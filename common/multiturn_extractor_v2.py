"""
Multi-Turn Extractor V2 - Bank-Agnostic Column-by-Column Extraction

Extracts bank statement transactions using:
1. Structure detection (identifies actual column headers)
2. Column-by-column focused extraction
3. Validation and cross-checking

Improvements over V1:
- Uses actual column headers from the document (bank-agnostic)
- Integrates with VisionLanguageModel and PromptRegistry
- Handles both flat and date-grouped table structures
- Uses "NOT_FOUND" for empty cells (consistent with extraction standards)

Usage:
    >>> from common.langchain_llm import VisionLanguageModel
    >>> from common.config import get_yaml_config
    >>>
    >>> # Load model and config
    >>> model, processor = load_llama_model(...)
    >>> llm = VisionLanguageModel(model=model, processor=processor)
    >>> config = get_yaml_config()
    >>>
    >>> # Extract bank statement
    >>> extractor = MultiTurnExtractorV2(llm=llm, config=config)
    >>> result = extractor.extract_bank_statement("statement.png")
    >>>
    >>> print(f"Extracted {result.row_count} transactions")
    >>> print(f"Structure: {result.structure.structure_type}")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from rich.console import Console

console = Console()


@dataclass
class TableStructure:
    """Detected table structure information."""

    structure_type: str  # "flat" or "date_grouped"
    column_headers: List[str]  # Actual headers from the image
    estimated_rows: int
    date_column: Optional[str] = None  # Mapped semantic columns
    description_column: Optional[str] = None
    debit_column: Optional[str] = None
    credit_column: Optional[str] = None
    balance_column: Optional[str] = None


@dataclass
class MultiTurnResult:
    """Result from multi-turn extraction."""

    structure: TableStructure
    dates: List[str]
    descriptions: List[str]
    debits: List[str]
    credits: List[str]
    balances: List[str]
    row_count: int
    validation_passed: bool
    validation_errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility with existing code."""
        return {
            "TRANSACTION_DATES": " | ".join(self.dates),
            "LINE_ITEM_DESCRIPTIONS": " | ".join(self.descriptions),
            "TRANSACTION_AMOUNTS_PAID": " | ".join(
                [d for d in self.debits if d != "NOT_FOUND"]
            ),
            "TRANSACTION_AMOUNTS_RECEIVED": " | ".join(
                [c for c in self.credits if c != "NOT_FOUND"]
            ),
            "_row_count": self.row_count,
            "_structure_type": self.structure.structure_type,
            "_validation_passed": self.validation_passed,
            "_validation_errors": self.validation_errors,
        }


class MultiTurnExtractorV2:
    """
    Multi-turn bank statement extractor using 3-turn approach.

    Architecture (NEW):
        Turn 0: Extract FULL markdown table + detect structure (single OCR)
        Turn 1: Filter to 3 columns (date, description, debit) - keep ALL rows
        Turn 2: Remove empty debit rows (withdrawals only)

    This approach avoids debit/credit confusion by:
        1. Extracting the complete table only once (no re-interpretation)
        2. Using text filtering for column selection
        3. Using empty-value filtering for row selection

    Example:
        >>> extractor = MultiTurnExtractorV2(llm=vision_llm, config=yaml_config)
        >>> markdown_table = extractor.extract_bank_statement("statement.png")
        >>> print(markdown_table)  # 3-column markdown (withdrawals only)
    """

    def __init__(self, llm: Any, config: Any):
        """
        Initialize multi-turn extractor.

        Args:
            llm: VisionLanguageModel instance
            config: YAMLConfigLoader instance
        """
        self.llm = llm
        self.config = config

    def extract_bank_statement(
        self, image_path: str | Path
    ) -> str:
        """
        Extract bank statement using 3-turn approach.

        Turn 0: Extract FULL markdown table + detect structure
        Turn 1: Filter to 3 columns (date, description, debit) - keep ALL rows
        Turn 2: Remove empty debit rows (withdrawals only)

        Args:
            image_path: Path to bank statement image

        Returns:
            Markdown table string with 3 columns (withdrawals only)

        Example:
            >>> markdown_table = extractor.extract_bank_statement("statement.png")
            >>> print(markdown_table)
        """
        image_path = Path(image_path)

        console.print(f"\n[bold cyan]Multi-Turn Extraction (3-Turn):[/bold cyan] {image_path.name}")

        # Turn 0: Extract FULL markdown table + detect structure
        console.print("[cyan]Turn 0:[/cyan] Extracting full markdown table + detecting structure...")
        structure, turn0_response = self._detect_structure(image_path)

        console.print(f"  Structure: [green]{structure.structure_type}[/green]")
        console.print(f"  Columns: [green]{' | '.join(structure.column_headers)}[/green]")
        console.print(f"  Estimated rows: [green]{structure.estimated_rows}[/green]")

        # Map column headers to semantic types
        structure = self._map_column_headers(structure)

        # Debug: Show mappings
        console.print("  [yellow]Mappings:[/yellow]")
        console.print(f"    Date → {structure.date_column}")
        console.print(f"    Description → {structure.description_column}")
        console.print(f"    Debit → {structure.debit_column}")

        # Turn 1: Filter to 3 columns (keep ALL rows)
        console.print("\n[cyan]Turn 1:[/cyan] Filtering to 3 columns (all rows)...")
        console.print(f"  Columns: [green]{structure.date_column} | {structure.description_column} | {structure.debit_column}[/green]")
        console.print("  [dim]Using Turn 0 response for conversation context...[/dim]")

        turn1_table = self._turn1_extract_3columns(
            image_path, structure, turn0_response
        )

        # DEBUG: Show Turn 1 response
        console.print("\n[yellow]DEBUG - Turn 1 Full Response:[/yellow]")
        console.print("[dim]" + "=" * 80 + "[/dim]")
        console.print(turn1_table)
        console.print("[dim]" + "=" * 80 + "[/dim]\n")

        # Turn 2: Remove rows with "NOT_FOUND" in debit column (withdrawals only)
        console.print("\n[cyan]Turn 2:[/cyan] Filtering to withdrawals only...")
        console.print(f"  Removing rows where '{structure.debit_column}' is \"NOT_FOUND\"...")
        console.print("  [dim]Using Turn 0 and Turn 1 responses for conversation context...[/dim]")

        markdown_table = self._turn2_filter_debits(
            image_path, structure, turn0_response, turn1_table
        )

        console.print("\n[bold green]✅ Extraction complete (3 turns)[/bold green]")

        return markdown_table

    def extract_bank_statement_OLD(
        self, image_path: str | Path
    ) -> MultiTurnResult:
        """OLD MULTI-COLUMN VERSION - DEPRECATED"""
        image_path = Path(image_path)

        console.print(f"\n[bold cyan]Multi-Turn Extraction:[/bold cyan] {image_path.name}")

        # Turn 0: Detect structure and column headers
        console.print("[cyan]Turn 0:[/cyan] Detecting table structure...")
        structure = self._detect_structure(image_path)

        console.print(f"  Structure: [green]{structure.structure_type}[/green]")
        console.print(f"  Columns: [green]{' | '.join(structure.column_headers)}[/green]")
        console.print(f"  Estimated rows: [green]{structure.estimated_rows}[/green]")

        # Map column headers to semantic types
        structure = self._map_column_headers(structure)

        # Debug: Show mappings
        console.print("  [yellow]Mappings:[/yellow]")
        console.print(f"    Date → {structure.date_column}")
        console.print(f"    Description → {structure.description_column}")
        console.print(f"    Debit → {structure.debit_column}")
        console.print(f"    Credit → {structure.credit_column}")
        console.print(f"    Balance → {structure.balance_column}")

        # Turn 1: Extract dates
        console.print(f"\n[cyan]Turn 1:[/cyan] Extracting '{structure.date_column}' column...")
        dates = self._extract_date_column(image_path, structure)
        console.print(f"  Extracted: [green]{len(dates)} dates[/green]")

        # Turn 2: Extract descriptions
        console.print(f"\n[cyan]Turn 2:[/cyan] Extracting '{structure.description_column}' column...")
        descriptions = self._extract_column(
            image_path,
            column_name=structure.description_column,
            template_key="column_extraction_template",
            additional_context="Extract the full transaction description as shown.",
        )
        console.print(f"  Extracted: [green]{len(descriptions)} descriptions[/green]")

        # Turn 3: Extract debits
        console.print(f"\n[cyan]Turn 3:[/cyan] Extracting '{structure.debit_column}' column...")
        debits = self._extract_column(
            image_path,
            column_name=structure.debit_column,
            template_key="column_extraction_template",
            additional_context="This column contains amounts that REDUCE the balance (money OUT).",
        )
        console.print(f"  Extracted: [green]{len(debits)} debit values[/green]")

        # Turn 4: Extract credits
        console.print(f"\n[cyan]Turn 4:[/cyan] Extracting '{structure.credit_column}' column...")
        credits = self._extract_column(
            image_path,
            column_name=structure.credit_column,
            template_key="column_extraction_template",
            additional_context="This column contains amounts that INCREASE the balance (money IN).",
        )
        console.print(f"  Extracted: [green]{len(credits)} credit values[/green]")

        # Turn 5: Extract balances (for validation)
        console.print(f"\n[cyan]Turn 5:[/cyan] Extracting '{structure.balance_column}' column...")
        balances = self._extract_column(
            image_path,
            column_name=structure.balance_column,
            template_key="column_extraction_template",
            additional_context="Extract the balance amount exactly as shown (including 'CR' notation if present).",
        )
        console.print(f"  Extracted: [green]{len(balances)} balance values[/green]")

        # Validate alignment
        console.print("\n[cyan]Validating:[/cyan] Column alignment...")
        validation_errors = self._validate_columns(
            dates, descriptions, debits, credits, balances
        )

        if validation_errors:
            console.print(f"  [yellow]⚠️  {len(validation_errors)} validation warnings[/yellow]")
            for error in validation_errors:
                console.print(f"    • {error}")
        else:
            console.print("  [green]✅ All columns aligned[/green]")

        result = MultiTurnResult(
            structure=structure,
            dates=dates,
            descriptions=descriptions,
            debits=debits,
            credits=credits,
            balances=balances,
            row_count=len(dates),
            validation_passed=len(validation_errors) == 0,
            validation_errors=validation_errors,
        )

        console.print(f"\n[bold green]✅ Extraction complete:[/bold green] {result.row_count} transactions")

        return result

    def _detect_structure(self, image_path: Path) -> tuple[TableStructure, str]:
        """
        Detect table structure and extract column headers.

        Args:
            image_path: Path to bank statement image

        Returns:
            tuple: (TableStructure, raw_response_text)
        """
        # Load structure detection prompt
        prompt_template = self.config.get_prompt_template("structure_detection_template")

        # Build message with image
        image = Image.open(image_path).convert("RGB")
        from langchain_core.messages import HumanMessage

        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": str(image_path)}},
                {"type": "text", "text": prompt_template},
            ]
        )

        # Get response
        response = self.llm.invoke([message])

        # DEBUG: Show raw Turn 0 response (full table)
        console.print("\n[yellow]DEBUG - Turn 0 Raw Response (FULL):[/yellow]")
        console.print("[dim]" + "=" * 80 + "[/dim]")
        console.print(response.content)  # Show complete response
        console.print("[dim]" + "=" * 80 + "[/dim]\n")

        # Parse response
        structure = self._parse_structure_response(response.content)

        # Return both structure and raw response for conversation history
        return structure, response.content

    def _parse_structure_response(self, response_text: str) -> TableStructure:
        """
        Parse structure detection response.

        Handles both formats:
        1. Plain: STRUCTURE_TYPE: flat, COLUMN_HEADERS: Date | Description | ...
        2. Markdown: **Structure Type:** flat, * Date, * Description, ...

        Args:
            response_text: Raw LLM output

        Returns:
            TableStructure instance
        """
        lines = response_text.strip().split("\n")

        structure_type = "flat"
        column_headers = []
        estimated_rows = 0
        in_headers_section = False

        for line in lines:
            line = line.strip()

            # Remove markdown bold formatting for easier parsing
            line_lower = line.lower().replace("*", "").strip()

            # Structure Type
            if "structure type:" in line_lower or line.startswith("STRUCTURE_TYPE:"):
                structure_type = line.split(":", 1)[1].replace("*", "").strip()

            # Column Headers (markdown bullet format)
            elif "column headers:" in line_lower or line.startswith("COLUMN_HEADERS:"):
                # Check if pipe-separated format on same line
                if "|" in line:
                    headers_str = line.split(":", 1)[1].strip()
                    column_headers = [h.strip() for h in headers_str.split("|")]
                else:
                    # Markdown bullet format follows on next lines
                    in_headers_section = True
                continue

            # Parse bullet point headers
            elif in_headers_section:
                if not line:
                    # Skip empty lines within headers section
                    continue
                elif line.startswith("*") and not line.startswith("**"):
                    # Bullet point header (but not markdown bold)
                    header = line[1:].strip()
                    column_headers.append(header)
                elif line.startswith("**"):
                    # End of headers section (new markdown section)
                    in_headers_section = False

            # Estimated Rows
            elif "estimated rows:" in line_lower or line.startswith("ESTIMATED_ROWS:"):
                try:
                    num_str = line.split(":", 1)[1].replace("*", "").strip()
                    estimated_rows = int(num_str)
                except (ValueError, IndexError):
                    estimated_rows = 0

        return TableStructure(
            structure_type=structure_type,
            column_headers=column_headers,
            estimated_rows=estimated_rows,
        )

    def _map_column_headers(self, structure: TableStructure) -> TableStructure:
        """
        Map detected column headers to semantic types.

        Args:
            structure: TableStructure with column_headers

        Returns:
            Updated TableStructure with semantic mappings
        """
        headers_lower = [h.lower() for h in structure.column_headers]

        # Map date column
        for i, header in enumerate(headers_lower):
            if "date" in header:
                structure.date_column = structure.column_headers[i]
                break

        # Map description column
        for i, header in enumerate(headers_lower):
            if any(
                keyword in header
                for keyword in ["description", "details", "transaction", "particulars"]
            ):
                structure.description_column = structure.column_headers[i]
                break

        # Map debit column
        for i, header in enumerate(headers_lower):
            if any(
                keyword in header
                for keyword in ["debit", "withdrawal", "withdrawals", "money out", "dr"]
            ):
                structure.debit_column = structure.column_headers[i]
                break

        # Map credit column
        for i, header in enumerate(headers_lower):
            if any(
                keyword in header
                for keyword in ["credit", "deposit", "deposits", "money in", "cr"]
            ):
                structure.credit_column = structure.column_headers[i]
                break

        # Map balance column
        for i, header in enumerate(headers_lower):
            if "balance" in header:
                structure.balance_column = structure.column_headers[i]
                break

        return structure

    def _extract_date_column(
        self, image_path: Path, structure: TableStructure
    ) -> List[str]:
        """
        Extract date column using specialized date template.

        Args:
            image_path: Path to image
            structure: Table structure info

        Returns:
            List of extracted dates
        """
        return self._extract_column(
            image_path,
            column_name=structure.date_column,
            template_key="date_column_extraction_template",
            additional_context="",
        )

    def _extract_column(
        self,
        image_path: Path,
        column_name: str,
        template_key: str,
        additional_context: str = "",
    ) -> List[str]:
        """
        Extract a single column from the bank statement.

        Args:
            image_path: Path to image
            column_name: Name of column (from structure detection)
            template_key: YAML template key
            additional_context: Extra instructions

        Returns:
            List of extracted values
        """
        # Load prompt template
        prompt_template = self.config.get_prompt_template(template_key)

        # Format with column name and context
        prompt = prompt_template.format(
            column_name=column_name, additional_context=additional_context
        )

        # Build message with image
        from langchain_core.messages import HumanMessage

        message = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": str(image_path)}},
                {"type": "text", "text": prompt},
            ]
        )

        # Get response
        response = self.llm.invoke([message])

        # Parse response into list of values
        values = self._parse_column_response(response.content)

        return values

    def _turn1_extract_3columns(
        self, image_path: Path, structure: TableStructure, turn0_response: str
    ) -> str:
        """
        Turn 1: Filter full markdown table to 3 columns (keep all rows).

        Args:
            image_path: Path to image
            structure: Table structure with column mappings
            turn0_response: Raw response from Turn 0 (structure detection)

        Returns:
            Markdown table string with 3 columns (all rows)
        """
        # Load Turn 1 template (filter to 3 columns)
        prompt_template = self.config.get_prompt_template(
            "turn1_3column_template"
        )

        # Format with actual column names
        turn1_prompt = prompt_template.format(
            date_column=structure.date_column,
            description_column=structure.description_column,
            debit_column=structure.debit_column,
        )

        # Build conversation history with Turn 0
        from langchain_core.messages import AIMessage, HumanMessage

        # Get Turn 0 prompt for conversation history
        turn0_prompt = self.config.get_prompt_template("structure_detection_template")

        messages = [
            # Turn 0: Structure detection (image included here)
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": str(image_path)}},
                    {"type": "text", "text": turn0_prompt},
                ]
            ),
            AIMessage(content=turn0_response),
            # Turn 1: 3-column extraction (NO image - model remembers from Turn 0)
            HumanMessage(
                content=[
                    {"type": "text", "text": turn1_prompt},
                ]
            ),
        ]

        # Get response with conversation context
        response = self.llm.invoke(messages)

        # Clean up markdown formatting if present
        markdown_table = response.content.strip()

        # Remove code fence markers if present
        if markdown_table.startswith("```markdown"):
            markdown_table = markdown_table[len("```markdown") :].strip()
        elif markdown_table.startswith("```"):
            markdown_table = markdown_table[3:].strip()

        if markdown_table.endswith("```"):
            markdown_table = markdown_table[:-3].strip()

        return markdown_table

    def _turn2_filter_debits(
        self,
        image_path: Path,
        structure: TableStructure,
        turn0_response: str,
        turn1_response: str
    ) -> str:
        """
        Turn 2: Filter to withdrawals only (remove empty debit rows).

        Args:
            image_path: Path to image
            structure: Table structure with column mappings
            turn0_response: Raw response from Turn 0 (structure detection)
            turn1_response: Raw response from Turn 1 (3-column markdown table)

        Returns:
            Markdown table string with withdrawals only
        """
        # Load Turn 2 template (filter to withdrawals)
        prompt_template = self.config.get_prompt_template(
            "turn2_filter_debits_template"
        )

        # Format with actual column names
        turn2_prompt = prompt_template.format(
            date_column=structure.date_column,
            description_column=structure.description_column,
            debit_column=structure.debit_column,
        )

        # Build conversation history with Turn 0 and Turn 1
        from langchain_core.messages import AIMessage, HumanMessage

        # Get Turn 0 and Turn 1 prompts for conversation history
        turn0_prompt = self.config.get_prompt_template("structure_detection_template")
        turn1_prompt_template = self.config.get_prompt_template("turn1_3column_template")
        turn1_prompt = turn1_prompt_template.format(
            date_column=structure.date_column,
            description_column=structure.description_column,
            debit_column=structure.debit_column,
        )

        messages = [
            # Turn 0: Structure detection (image included here)
            HumanMessage(
                content=[
                    {"type": "image_url", "image_url": {"url": str(image_path)}},
                    {"type": "text", "text": turn0_prompt},
                ]
            ),
            AIMessage(content=turn0_response),
            # Turn 1: 3-column extraction (NO image - model remembers from Turn 0)
            HumanMessage(
                content=[
                    {"type": "text", "text": turn1_prompt},
                ]
            ),
            AIMessage(content=turn1_response),
            # Turn 2: Filter to withdrawals (NO image - model remembers from Turn 0)
            HumanMessage(
                content=[
                    {"type": "text", "text": turn2_prompt},
                ]
            ),
        ]

        # Get response with full conversation context
        response = self.llm.invoke(messages)

        # Clean up markdown formatting if present
        markdown_table = response.content.strip()

        # Remove code fence markers if present
        if markdown_table.startswith("```markdown"):
            markdown_table = markdown_table[len("```markdown") :].strip()
        elif markdown_table.startswith("```"):
            markdown_table = markdown_table[3:].strip()

        if markdown_table.endswith("```"):
            markdown_table = markdown_table[:-3].strip()

        return markdown_table

    def _parse_column_response(self, response_text: str) -> List[str]:
        """
        Parse column extraction response into list of values.

        Args:
            response_text: Raw LLM output

        Returns:
            List of extracted values (one per line)
        """
        lines = response_text.strip().split("\n")
        values = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip markdown headers
            if line.startswith("#"):
                continue

            # Skip explanation lines (contain colons but aren't values)
            if ":" in line and not any(c.isdigit() for c in line[:10]):
                continue

            # Skip numbered list markers
            line = line.lstrip("0123456789.-) ")

            values.append(line)

        return values

    def _validate_columns(
        self,
        dates: List[str],
        descriptions: List[str],
        debits: List[str],
        credits: List[str],
        balances: List[str],
    ) -> List[str]:
        """
        Validate that all columns have consistent row counts.

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
            max_count = max(counts.values())
            for col_name, count in counts.items():
                if count != max_count:
                    errors.append(
                        f"  {col_name}: {count} rows (expected {max_count})"
                    )

        return errors
