"""
Multi-Turn Extractor for Complex Bank Statements

Extracts complex tables by focusing on one column at a time across multiple LLM calls.
Reduces hallucination and column confusion for 5+ column bank statements.

Based on LMM_POC/single_column_multiturn_workflow.txt

Usage:
    >>> extractor = MultiTurnExtractor(llm=vision_model)
    >>> result = extractor.extract_bank_statement(image_path="complex_statement.png")
    >>> print(f"Extracted {len(result['dates'])} transactions")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class MultiTurnResult:
    """Result from multi-turn extraction."""

    dates: List[str]
    descriptions: List[str]
    debits: List[str]
    credits: List[str]
    balances: List[str]
    row_count: int
    validation_passed: bool
    validation_errors: List[str]


class MultiTurnExtractor:
    """
    Extracts complex bank statements using multiple focused passes.

    Each pass extracts a single column, reducing confusion and hallucination.

    Workflow:
        Turn 1: Extract Date column only
        Turn 2: Extract Description column only
        Turn 3: Extract Debit column only
        Turn 4: Extract Credit column only
        Turn 5: Extract Balance column only

    Example:
        >>> from langchain_llm import get_vision_llm
        >>> llm = get_vision_llm("llama-3.2-vision")
        >>> extractor = MultiTurnExtractor(llm=llm)
        >>>
        >>> result = extractor.extract_bank_statement(image_path="statement.png")
        >>> if result.validation_passed:
        ...     print(f"Successfully extracted {result.row_count} transactions")
        ...     for i in range(result.row_count):
        ...         print(f"{result.dates[i]} | {result.descriptions[i]} | {result.debits[i]}")
    """

    def __init__(self, llm: Any):
        """
        Initialize multi-turn extractor.

        Args:
            llm: Vision-language model (LangChain BaseChatModel)
        """
        self.llm = llm

    def extract_bank_statement(
        self,
        image_path: Optional[Path] = None,
        image_url: Optional[str] = None,
    ) -> MultiTurnResult:
        """
        Extract bank statement using multi-turn approach.

        Args:
            image_path: Path to bank statement image
            image_url: URL to bank statement image

        Returns:
            MultiTurnResult with all extracted columns

        Example:
            >>> result = extractor.extract_bank_statement(image_path="statement.png")
            >>> print(f"Extracted {result.row_count} rows")
        """
        if image_path is None and image_url is None:
            raise ValueError("Must provide either image_path or image_url")

        # Turn 1: Extract dates
        dates = self._extract_column(
            column_name="Date",
            column_description="date values",
            image_path=image_path,
            image_url=image_url,
        )

        # Turn 2: Extract descriptions
        descriptions = self._extract_column(
            column_name="Description",
            column_description="transaction descriptions",
            image_path=image_path,
            image_url=image_url,
            additional_instructions="If a description spans multiple lines, combine them.",
        )

        # Turn 3: Extract debits
        debits = self._extract_column(
            column_name="Debit",
            column_description="debit amounts",
            image_path=image_path,
            image_url=image_url,
            additional_instructions="If a cell is empty, write 'EMPTY'.",
        )

        # Turn 4: Extract credits
        credits = self._extract_column(
            column_name="Credit",
            column_description="credit amounts",
            image_path=image_path,
            image_url=image_url,
            additional_instructions="If a cell is empty, write 'EMPTY'. NEVER add 'CR' suffix.",
        )

        # Turn 5: Extract balances
        balances = self._extract_column(
            column_name="Balance",
            column_description="balance amounts",
            image_path=image_path,
            image_url=image_url,
            additional_instructions="Preserve 'CR' notation exactly as shown.",
        )

        # Validate alignment
        validation_errors = self._validate_alignment(
            dates, descriptions, debits, credits, balances
        )

        return MultiTurnResult(
            dates=dates,
            descriptions=descriptions,
            debits=debits,
            credits=credits,
            balances=balances,
            row_count=len(dates),
            validation_passed=len(validation_errors) == 0,
            validation_errors=validation_errors,
        )

    def _extract_column(
        self,
        column_name: str,
        column_description: str,
        image_path: Optional[Path],
        image_url: Optional[str],
        additional_instructions: str = "",
    ) -> List[str]:
        """
        Extract a single column from the bank statement.

        Args:
            column_name: Name of column (e.g., "Date", "Debit")
            column_description: Description for prompt
            image_path: Path to image
            image_url: URL to image
            additional_instructions: Extra instructions for this column

        Returns:
            List of extracted values
        """
        # Build column-specific prompt
        prompt = self._build_column_prompt(
            column_name, column_description, additional_instructions
        )

        # Build messages
        messages = self._build_messages(prompt, image_path, image_url)

        # Invoke LLM
        response = self.llm.invoke(messages)

        # Parse response
        values = self._parse_column_response(response.content)

        return values

    def _build_column_prompt(
        self,
        column_name: str,
        column_description: str,
        additional_instructions: str,
    ) -> str:
        """
        Build prompt for extracting a specific column.

        Args:
            column_name: Column name
            column_description: Description
            additional_instructions: Extra instructions

        Returns:
            Prompt string
        """
        prompt = f"""Look at the transaction table in this bank statement.

Find the column with the header "{column_name}".

Extract ONLY the {column_description} from this column, ignoring all other columns.

Important:
- Extract ONLY {column_description}
- Do NOT include values from other columns
- List one value per line
{additional_instructions}

Output format:
[First value]
[Second value]
[Third value]
..."""

        return prompt

    def _build_messages(
        self,
        prompt: str,
        image_path: Optional[Path],
        image_url: Optional[str],
    ) -> list:
        """Build messages for vision LLM."""
        system_msg = SystemMessage(
            content="You are a precise data extractor. Extract only the requested column."
        )

        if image_path:
            from PIL import Image

            image = Image.open(image_path)

            human_msg = HumanMessage(
                content=[
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            )
        elif image_url:
            human_msg = HumanMessage(
                content=[
                    {"type": "image_url", "image_url": image_url},
                    {"type": "text", "text": prompt}
                ]
            )
        else:
            raise ValueError("Must provide either image_path or image_url")

        return [system_msg, human_msg]

    def _parse_column_response(self, response_text: str) -> List[str]:
        """
        Parse LLM response into list of values.

        Args:
            response_text: Raw LLM output

        Returns:
            List of extracted values
        """
        lines = response_text.strip().split("\n")
        values = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip markdown formatting
            if line.startswith("#") or line.startswith("*"):
                continue

            # Skip explanation lines
            if ":" in line and not line[0].isdigit():
                continue

            values.append(line)

        return values

    def _validate_alignment(
        self,
        dates: List[str],
        descriptions: List[str],
        debits: List[str],
        credits: List[str],
        balances: List[str],
    ) -> List[str]:
        """
        Validate that all columns have the same row count.

        Args:
            dates: Date column values
            descriptions: Description column values
            debits: Debit column values
            credits: Credit column values
            balances: Balance column values

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        counts = {
            "dates": len(dates),
            "descriptions": len(descriptions),
            "debits": len(debits),
            "credits": len(credits),
            "balances": len(balances),
        }

        # Check if all counts match
        unique_counts = set(counts.values())

        if len(unique_counts) > 1:
            errors.append(f"Column count mismatch: {counts}")
            errors.append("All columns must have the same number of rows")

        return errors

    def extract_debit_only(
        self,
        image_path: Optional[Path] = None,
        image_url: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Extract only debit transactions (simplified 3-column extraction).

        Useful for taxpayer expense claims.

        Args:
            image_path: Path to bank statement image
            image_url: URL to bank statement image

        Returns:
            Dictionary with dates, descriptions, and amounts

        Example:
            >>> result = extractor.extract_debit_only(image_path="statement.png")
            >>> for i in range(len(result['dates'])):
            ...     print(f"{result['dates'][i]} | {result['descriptions'][i]} | {result['amounts'][i]}")
        """
        # Turn 1: Dates
        dates = self._extract_column(
            "Date", "date values", image_path, image_url
        )

        # Turn 2: Descriptions
        descriptions = self._extract_column(
            "Description", "transaction descriptions", image_path, image_url
        )

        # Turn 3: Debit amounts only
        debits = self._extract_column(
            "Debit",
            "debit amounts",
            image_path,
            image_url,
            additional_instructions="Extract ONLY debit/withdrawal amounts. Skip credit entries.",
        )

        return {
            "dates": dates,
            "descriptions": descriptions,
            "amounts": debits,
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def extract_complex_bank_statement(
    image_path: Path,
    llm: Any,
) -> MultiTurnResult:
    """
    Quick function to extract complex bank statement.

    Args:
        image_path: Path to bank statement image
        llm: Vision-language model

    Returns:
        MultiTurnResult with all columns

    Example:
        >>> from langchain_llm import get_vision_llm
        >>> llm = get_vision_llm("llama-3.2-vision")
        >>> result = extract_complex_bank_statement(
        ...     image_path=Path("complex_statement.png"),
        ...     llm=llm
        ... )
        >>> print(f"Extracted {result.row_count} transactions")
    """
    extractor = MultiTurnExtractor(llm=llm)
    return extractor.extract_bank_statement(image_path=image_path)
