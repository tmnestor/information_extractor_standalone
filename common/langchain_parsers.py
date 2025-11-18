"""
LangChain Output Parsers for Document Extraction

Provides Pydantic-based output parsers with self-healing capabilities.
Replaces ~400 lines of custom parsing code from extraction_parser.py.

Key Features:
- Pydantic validation with automatic type coercion
- Self-healing parser (retries with LLM on errors)
- JSON and plain text extraction fallback
- Markdown artifact cleaning
- Document-type-aware parsing
"""

import json
import re
from typing import Any, Dict, Optional

from langchain_core.output_parsers import (
    PydanticOutputParser,
)
from pydantic import ValidationError

from .extraction_schemas import (
    BaseExtractionSchema,
    get_schema_for_document_type,
)

# OutputFixingParser is optional (only for self-healing)
try:
    from langchain.output_parsers import OutputFixingParser
    HAS_OUTPUT_FIXING = True
except ImportError:
    HAS_OUTPUT_FIXING = False
    OutputFixingParser = None


class DocumentExtractionParser:
    """
    Main parser for document extraction with multiple fallback strategies.

    Provides:
    1. JSON extraction (if model returns JSON)
    2. Plain text field extraction (key: value pairs)
    3. Markdown artifact cleaning
    4. Pydantic validation
    5. Self-healing with LLM (optional)

    Usage:
        >>> parser = DocumentExtractionParser(document_type="invoice")
        >>> result = parser.parse(model_output)
        >>> print(result.TOTAL_AMOUNT)  # Typed access

        >>> # With self-healing
        >>> parser = DocumentExtractionParser(
        ...     document_type="invoice",
        ...     llm=llm,
        ...     enable_fixing=True
        ... )
        >>> result = parser.parse_with_fixing(model_output)
    """

    def __init__(
        self,
        document_type: str = "universal",
        llm: Optional[Any] = None,
        enable_fixing: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize parser for specific document type.

        Args:
            document_type: Type of document (invoice, receipt, bank_statement, universal)
            llm: LangChain LLM for self-healing (required if enable_fixing=True)
            enable_fixing: Enable self-healing parser
            verbose: Enable verbose logging
        """
        self.document_type = document_type.lower()
        self.llm = llm
        self.enable_fixing = enable_fixing
        self.verbose = verbose

        # Get appropriate Pydantic schema
        self.schema_class = get_schema_for_document_type(self.document_type)

        # Create base Pydantic parser
        self.base_parser = PydanticOutputParser(pydantic_object=self.schema_class)

        # Create self-healing parser if enabled
        self.fixing_parser = None
        if enable_fixing and llm:
            if HAS_OUTPUT_FIXING:
                self.fixing_parser = OutputFixingParser.from_llm(
                    parser=self.base_parser, llm=llm, max_retries=2
                )
            elif verbose:
                print("⚠️  OutputFixingParser not available - self-healing disabled")
                print("   Install with: pip install langchain[output_parsers]")

    def parse(self, output: str) -> BaseExtractionSchema:
        """
        Parse model output into Pydantic schema.

        Attempts multiple strategies:
        1. JSON extraction
        2. Plain text field extraction
        3. Pydantic validation

        Args:
            output: Raw model output string

        Returns:
            Validated Pydantic model instance

        Raises:
            OutputParserException: If parsing fails after all strategies
        """
        # Strategy 1: Try JSON extraction first
        try:
            extracted_dict = self._extract_json(output)
            if extracted_dict:
                return self.schema_class(**extracted_dict)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Strategy 2: Plain text field extraction
        try:
            extracted_dict = self._extract_plain_text(output)
            return self.schema_class(**extracted_dict)
        except ValidationError as e:
            if self.verbose:
                print(f"Validation error: {e}")
            # Try to proceed with partial data
            return self.schema_class(**extracted_dict)

    def parse_with_fixing(self, output: str) -> BaseExtractionSchema:
        """
        Parse with self-healing (requires LLM).

        If initial parse fails, uses LLM to correct the output and retry.

        Args:
            output: Raw model output string

        Returns:
            Validated Pydantic model instance

        Raises:
            ValueError: If fixing parser not enabled
            OutputParserException: If parsing fails after retries
        """
        if not self.fixing_parser:
            if not HAS_OUTPUT_FIXING:
                # OutputFixingParser not available, fall back to regular parsing
                if self.verbose:
                    print("⚠️  Self-healing not available, using regular parsing")
                return self.parse(output)
            else:
                raise ValueError(
                    "Self-healing parser not enabled. "
                    "Initialize with llm and enable_fixing=True"
                )

        return self.fixing_parser.parse(output)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from model output.

        Handles:
        - Markdown code blocks (```json ... ```)
        - Inline JSON objects
        - Conversation artifacts before/after JSON

        Args:
            text: Model output text

        Returns:
            Extracted JSON dict or None
        """
        # Remove markdown code blocks
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        # Find JSON object
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _extract_plain_text(self, text: str) -> Dict[str, Any]:
        """
        Extract fields from plain text output (key: value pairs).

        Handles:
        - "FIELD_NAME: value" format
        - Multiline values
        - Pipe-separated lists
        - Markdown artifacts

        Args:
            text: Model output text

        Returns:
            Dictionary of extracted fields
        """
        # Clean markdown and conversation artifacts
        text = self._clean_text(text)

        # Get expected fields for this document type
        expected_fields = self._get_expected_fields()

        extracted = {}

        # Extract each field
        for field in expected_fields:
            # Pattern: FIELD_NAME: value
            pattern = rf"{field}:\s*(.+?)(?=\n[A-Z_]+:|$)"
            match = re.search(pattern, text, re.DOTALL)

            if match:
                value = match.group(1).strip()
                # Clean brackets/quotes
                value = re.sub(r"^\[|\]$", "", value)
                value = value.strip('"\'')

                extracted[field] = value if value else "NOT_FOUND"
            else:
                extracted[field] = "NOT_FOUND"

        return extracted

    def _clean_text(self, text: str) -> str:
        """
        Clean conversational artifacts from model output.

        Removes:
        - Markdown formatting
        - Conversation responses
        - Explanatory text

        Args:
            text: Raw model output

        Returns:
            Cleaned text
        """
        # Remove common conversation starters
        conversation_patterns = [
            r"^(Here is|Here are|I've extracted|Based on).*?\n",
            r"^(The document|This appears).*?\n",
            r"```[a-z]*\n?",  # Markdown code blocks
            r"\n```\s*$",
        ]

        for pattern in conversation_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

        return text.strip()

    def _get_expected_fields(self) -> list[str]:
        """
        Get list of expected fields for current document type.

        Returns:
            List of field names
        """
        # Get fields from Pydantic model
        return list(self.schema_class.model_fields.keys())

    def get_format_instructions(self) -> str:
        """
        Get format instructions for the model prompt.

        Returns:
            String containing formatting instructions
        """
        return self.base_parser.get_format_instructions()


class DocumentTypeParser:
    """
    Simple parser for document type detection.

    Extracts document type from model output, handling various formats.

    Usage:
        >>> parser = DocumentTypeParser()
        >>> doc_type = parser.parse("DOCUMENT_TYPE: INVOICE")
        >>> print(doc_type)  # "invoice"
    """

    VALID_TYPES = ["invoice", "receipt", "bank_statement", "statement"]

    def parse(self, output: str) -> str:
        """
        Parse document type from model output.

        Args:
            output: Model output containing document type

        Returns:
            Normalized document type (lowercase)
        """
        # Clean output
        output_lower = output.lower().strip()

        # Try direct extraction
        for doc_type in self.VALID_TYPES:
            if doc_type in output_lower:
                # Normalize
                if "statement" in doc_type and "bank" not in doc_type:
                    return "bank_statement"
                return doc_type

        # Try pattern matching
        match = re.search(r"(?:document_type|type):\s*([a-zA-Z_\s]+)", output, re.IGNORECASE)
        if match:
            extracted_type = match.group(1).strip().lower()
            extracted_type = extracted_type.replace(" ", "_")

            # Validate against known types
            if extracted_type in self.VALID_TYPES:
                return extracted_type

            # Partial match
            for doc_type in self.VALID_TYPES:
                if doc_type in extracted_type:
                    return doc_type

        # Default fallback
        return "unknown"


# ============================================================================
# Parser Factory - Create parsers for different document types
# ============================================================================

class ParserFactory:
    """
    Factory for creating document-specific parsers.

    Simplifies parser creation and management.

    Usage:
        >>> factory = ParserFactory(llm=llm, enable_fixing=True)
        >>> invoice_parser = factory.get_parser("invoice")
        >>> result = invoice_parser.parse(model_output)
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        enable_fixing: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize parser factory.

        Args:
            llm: LangChain LLM for self-healing
            enable_fixing: Enable self-healing parsers
            verbose: Enable verbose logging
        """
        self.llm = llm
        self.enable_fixing = enable_fixing
        self.verbose = verbose

        # Cache parsers
        self._parsers: Dict[str, DocumentExtractionParser] = {}

    def get_parser(self, document_type: str) -> DocumentExtractionParser:
        """
        Get or create parser for document type.

        Args:
            document_type: Type of document

        Returns:
            DocumentExtractionParser instance
        """
        if document_type not in self._parsers:
            self._parsers[document_type] = DocumentExtractionParser(
                document_type=document_type,
                llm=self.llm,
                enable_fixing=self.enable_fixing,
                verbose=self.verbose,
            )

        return self._parsers[document_type]

    def get_type_parser(self) -> DocumentTypeParser:
        """
        Get document type detection parser.

        Returns:
            DocumentTypeParser instance
        """
        return DocumentTypeParser()


# ============================================================================
# Convenience Functions
# ============================================================================

def parse_extraction(
    output: str,
    document_type: str = "universal",
    llm: Optional[Any] = None,
    enable_fixing: bool = False,
) -> BaseExtractionSchema:
    """
    Quick parse function for simple use cases.

    Args:
        output: Model output to parse
        document_type: Type of document
        llm: LangChain LLM for self-healing
        enable_fixing: Enable self-healing

    Returns:
        Parsed Pydantic model

    Example:
        >>> result = parse_extraction(
        ...     model_output,
        ...     document_type="invoice",
        ...     enable_fixing=True,
        ...     llm=llm
        ... )
        >>> print(result.TOTAL_AMOUNT)
    """
    parser = DocumentExtractionParser(
        document_type=document_type, llm=llm, enable_fixing=enable_fixing
    )

    if enable_fixing and llm:
        return parser.parse_with_fixing(output)
    else:
        return parser.parse(output)


def parse_document_type(output: str) -> str:
    """
    Quick parse function for document type detection.

    Args:
        output: Model output containing document type

    Returns:
        Normalized document type

    Example:
        >>> doc_type = parse_document_type("DOCUMENT_TYPE: INVOICE")
        >>> print(doc_type)  # "invoice"
    """
    parser = DocumentTypeParser()
    return parser.parse(output)
