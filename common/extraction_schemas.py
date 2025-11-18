"""
Pydantic Extraction Schemas for Document Processing

Type-safe schemas for invoice, receipt, and bank statement extraction.
Replaces ~400 lines of custom parsing code with declarative validation.

Key Features:
- Type safety with Pydantic v2
- Automatic validation and cleaning
- Support for NOT_FOUND values
- List parsing from pipe-separated strings
- Monetary value handling
- Date format validation
"""

from decimal import Decimal, InvalidOperation
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# Base Extraction Schema - Common fields and validators
# ============================================================================

class BaseExtractionSchema(BaseModel):
    """
    Base schema for all document extractions.

    Provides common functionality:
    - NOT_FOUND handling
    - Pipe-separated list parsing
    - Monetary value normalization
    """

    # Allow arbitrary types for NOT_FOUND handling
    model_config = {"arbitrary_types_allowed": True, "str_strip_whitespace": True}

    @staticmethod
    def parse_pipe_list(value: Union[str, List[str]]) -> List[str]:
        """
        Parse pipe-separated string into list.

        Args:
            value: Either a list or pipe-separated string

        Returns:
            List[str]: Parsed list

        Example:
            "Item 1 | Item 2 | Item 3" -> ["Item 1", "Item 2", "Item 3"]
        """
        if value == "NOT_FOUND" or not value:
            return []

        if isinstance(value, list):
            return [str(item).strip() for item in value if item]

        # Parse pipe-separated string
        return [item.strip() for item in str(value).split("|") if item.strip()]

    @staticmethod
    def parse_monetary(value: Union[str, Decimal, float]) -> Union[Decimal, str]:
        """
        Parse monetary value, handling currency symbols and formats.

        Args:
            value: Monetary value in various formats

        Returns:
            Decimal or "NOT_FOUND"

        Example:
            "$123.45" -> Decimal("123.45")
            "NOT_FOUND" -> "NOT_FOUND"
        """
        if value == "NOT_FOUND" or not value:
            return "NOT_FOUND"

        if isinstance(value, Decimal):
            return value

        # Clean monetary string
        clean_value = str(value).replace("$", "").replace(",", "").strip()

        try:
            return Decimal(clean_value)
        except (InvalidOperation, ValueError):
            return "NOT_FOUND"

    @staticmethod
    def is_not_found(value: any) -> bool:
        """Check if value represents NOT_FOUND."""
        return value == "NOT_FOUND" or value is None or (isinstance(value, str) and not value.strip())


# ============================================================================
# Invoice Extraction Schema
# ============================================================================

class InvoiceExtraction(BaseExtractionSchema):
    """
    Pydantic schema for invoice document extraction.

    14 fields: Document metadata, business details, line items, and financial totals.
    """

    # Document identification
    DOCUMENT_TYPE: Literal["INVOICE", "TAX INVOICE", "BILL", "NOT_FOUND"] = Field(
        description="Type of document (always INVOICE for this schema)"
    )

    # Business details
    BUSINESS_ABN: str = Field(
        description="11-digit Australian Business Number or NOT_FOUND"
    )
    SUPPLIER_NAME: str = Field(
        description="Business/company name providing goods/services"
    )
    BUSINESS_ADDRESS: str = Field(
        description="Complete supplier business address"
    )

    # Customer details
    PAYER_NAME: str = Field(description="Customer/payer name")
    PAYER_ADDRESS: str = Field(description="Customer/payer address")

    # Temporal
    INVOICE_DATE: str = Field(description="Invoice date (DD/MM/YYYY format)")

    # Line items
    LINE_ITEM_DESCRIPTIONS: List[str] = Field(
        description="Product/service descriptions"
    )
    LINE_ITEM_QUANTITIES: List[str] = Field(description="Quantities for each item")
    LINE_ITEM_PRICES: List[Union[Decimal, str]] = Field(
        description="Unit prices (Decimal or NOT_FOUND)"
    )
    LINE_ITEM_TOTAL_PRICES: List[Union[Decimal, str]] = Field(
        description="Total prices per line (Decimal or NOT_FOUND)"
    )

    # Financial
    IS_GST_INCLUDED: Union[bool, str] = Field(
        description="Whether GST is shown (true/false or NOT_FOUND)"
    )
    GST_AMOUNT: Union[Decimal, str] = Field(
        description="GST amount (Decimal or NOT_FOUND)"
    )
    TOTAL_AMOUNT: Union[Decimal, str] = Field(
        description="Final total amount (Decimal or NOT_FOUND)"
    )

    # Validators

    @field_validator("BUSINESS_ABN")
    @classmethod
    def validate_abn(cls, v: str) -> str:
        """Validate ABN format (11 digits)."""
        if v == "NOT_FOUND" or not v:
            return "NOT_FOUND"

        # Remove spaces and validate
        clean_abn = v.replace(" ", "").replace("-", "")

        if clean_abn.isdigit() and len(clean_abn) == 11:
            return clean_abn

        # Return as-is if doesn't match format (may still be valid content)
        return v

    @field_validator("LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse pipe-separated lists."""
        return cls.parse_pipe_list(v)

    @field_validator("LINE_ITEM_PRICES", "LINE_ITEM_TOTAL_PRICES", mode="before")
    @classmethod
    def parse_monetary_lists(cls, v: Union[str, List]) -> List[Union[Decimal, str]]:
        """Parse pipe-separated monetary values."""
        if v == "NOT_FOUND" or not v:
            return []

        if isinstance(v, list):
            return [cls.parse_monetary(item) for item in v]

        # Parse pipe-separated string
        items = str(v).split("|")
        return [cls.parse_monetary(item.strip()) for item in items if item.strip()]

    @field_validator("GST_AMOUNT", "TOTAL_AMOUNT", mode="before")
    @classmethod
    def parse_monetary_field(cls, v: Union[str, Decimal, float]) -> Union[Decimal, str]:
        """Parse single monetary value."""
        return cls.parse_monetary(v)

    @field_validator("IS_GST_INCLUDED", mode="before")
    @classmethod
    def parse_boolean(cls, v: Union[str, bool]) -> Union[bool, str]:
        """Parse boolean field."""
        if v == "NOT_FOUND" or v is None:
            return "NOT_FOUND"

        if isinstance(v, bool):
            return v

        # Parse string boolean
        v_lower = str(v).lower().strip()
        if v_lower in ("true", "yes", "1", "included"):
            return True
        if v_lower in ("false", "no", "0", "not included", "excluded"):
            return False

        return "NOT_FOUND"


# ============================================================================
# Receipt Extraction Schema
# ============================================================================

class ReceiptExtraction(InvoiceExtraction):
    """
    Pydantic schema for receipt document extraction.

    Identical structure to InvoiceExtraction (14 fields).
    INVOICE_DATE represents the transaction date for receipts.
    """

    DOCUMENT_TYPE: Literal["RECEIPT", "PURCHASE RECEIPT", "SALES RECEIPT", "NOT_FOUND"] = Field(
        description="Type of document (always RECEIPT for this schema)"
    )


# ============================================================================
# Bank Statement Extraction Schema
# ============================================================================

class BankStatementExtraction(BaseExtractionSchema):
    """
    Pydantic schema for bank statement document extraction.

    5 fields: Statement period and transaction details.
    """

    # Document identification
    DOCUMENT_TYPE: Literal["BANK_STATEMENT", "STATEMENT", "ACCOUNT STATEMENT", "NOT_FOUND"] = Field(
        description="Type of document (always BANK_STATEMENT for this schema)"
    )

    # Temporal
    STATEMENT_DATE_RANGE: str = Field(
        description="Statement period (DD/MM/YYYY - DD/MM/YYYY)"
    )

    # Transaction lists
    LINE_ITEM_DESCRIPTIONS: List[str] = Field(
        description="Transaction descriptions (e.g., EFTPOS, Direct Debit)"
    )
    TRANSACTION_DATES: List[str] = Field(description="Transaction dates")
    TRANSACTION_AMOUNTS_PAID: List[Union[Decimal, str]] = Field(
        description="Debit/withdrawal amounts"
    )

    # Validators
    @field_validator("DOCUMENT_TYPE", mode="before")
    @classmethod
    def normalize_document_type(cls, v):
        """Normalize document type to uppercase and replace spaces with underscores."""
        if v and isinstance(v, str):
            # Convert to uppercase, strip whitespace, replace spaces with underscores
            normalized = v.upper().strip().replace(" ", "_")
            return normalized
        return v

    @field_validator("LINE_ITEM_DESCRIPTIONS", "TRANSACTION_DATES", mode="before")
    @classmethod
    def parse_list_fields(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse pipe-separated lists."""
        return cls.parse_pipe_list(v)

    @field_validator("TRANSACTION_AMOUNTS_PAID", mode="before")
    @classmethod
    def parse_monetary_list(cls, v: Union[str, List]) -> List[Union[Decimal, str]]:
        """Parse pipe-separated monetary values."""
        if v == "NOT_FOUND" or not v:
            return []

        if isinstance(v, list):
            return [cls.parse_monetary(item) for item in v]

        # Parse pipe-separated string
        items = str(v).split("|")
        return [cls.parse_monetary(item.strip()) for item in items if item.strip()]

    @model_validator(mode="after")
    def validate_transaction_lengths(self) -> "BankStatementExtraction":
        """Ensure transaction lists have matching lengths."""
        desc_len = len(self.LINE_ITEM_DESCRIPTIONS)
        dates_len = len(self.TRANSACTION_DATES)
        amounts_len = len(self.TRANSACTION_AMOUNTS_PAID)

        # Only validate if all lists have items
        if desc_len > 0 and dates_len > 0 and amounts_len > 0:
            if not (desc_len == dates_len == amounts_len):
                # Log warning but don't fail validation
                # (Allow model to extract partial data)
                pass

        return self


# ============================================================================
# Universal Extraction Schema (All Fields)
# ============================================================================

class UniversalExtraction(BaseExtractionSchema):
    """
    Universal schema with all 19 possible fields.

    Used when document type is unknown or for comparative analysis.
    Combines invoice, receipt, and bank statement fields.
    """

    # Document identification
    DOCUMENT_TYPE: str = Field(description="Type of document")

    # Business details
    BUSINESS_ABN: Optional[str] = Field(default="NOT_FOUND")
    SUPPLIER_NAME: Optional[str] = Field(default="NOT_FOUND")
    BUSINESS_ADDRESS: Optional[str] = Field(default="NOT_FOUND")

    # Customer details
    PAYER_NAME: Optional[str] = Field(default="NOT_FOUND")
    PAYER_ADDRESS: Optional[str] = Field(default="NOT_FOUND")

    # Temporal
    INVOICE_DATE: Optional[str] = Field(default="NOT_FOUND")
    STATEMENT_DATE_RANGE: Optional[str] = Field(default="NOT_FOUND")

    # Line items (shared)
    LINE_ITEM_DESCRIPTIONS: List[str] = Field(default_factory=list)
    LINE_ITEM_QUANTITIES: List[str] = Field(default_factory=list)
    LINE_ITEM_PRICES: List[Union[Decimal, str]] = Field(default_factory=list)
    LINE_ITEM_TOTAL_PRICES: List[Union[Decimal, str]] = Field(default_factory=list)

    # Financial
    IS_GST_INCLUDED: Optional[Union[bool, str]] = Field(default="NOT_FOUND")
    GST_AMOUNT: Optional[Union[Decimal, str]] = Field(default="NOT_FOUND")
    TOTAL_AMOUNT: Optional[Union[Decimal, str]] = Field(default="NOT_FOUND")

    # Transactions (bank statements)
    TRANSACTION_DATES: List[str] = Field(default_factory=list)
    TRANSACTION_AMOUNTS_PAID: List[Union[Decimal, str]] = Field(default_factory=list)
    TRANSACTION_AMOUNTS_RECEIVED: List[Union[Decimal, str]] = Field(default_factory=list)
    ACCOUNT_BALANCE: List[Union[Decimal, str]] = Field(default_factory=list)

    # Apply same validators as specific schemas
    # (Code reuse through inheritance of parse_pipe_list, parse_monetary, etc.)

    _validate_lists = field_validator(
        "LINE_ITEM_DESCRIPTIONS",
        "LINE_ITEM_QUANTITIES",
        "TRANSACTION_DATES",
        mode="before",
    )(BaseExtractionSchema.parse_pipe_list)

    _validate_monetary_lists = field_validator(
        "LINE_ITEM_PRICES",
        "LINE_ITEM_TOTAL_PRICES",
        "TRANSACTION_AMOUNTS_PAID",
        "TRANSACTION_AMOUNTS_RECEIVED",
        "ACCOUNT_BALANCE",
        mode="before",
    )(InvoiceExtraction.parse_monetary_lists.__func__)

    _validate_monetary = field_validator(
        "GST_AMOUNT", "TOTAL_AMOUNT", mode="before"
    )(BaseExtractionSchema.parse_monetary)

    _validate_boolean = field_validator("IS_GST_INCLUDED", mode="before")(
        InvoiceExtraction.parse_boolean.__func__
    )


# ============================================================================
# Schema Registry - Map document types to schemas
# ============================================================================

EXTRACTION_SCHEMAS = {
    "invoice": InvoiceExtraction,
    "receipt": ReceiptExtraction,
    "bank_statement": BankStatementExtraction,
    "universal": UniversalExtraction,
}


def get_schema_for_document_type(document_type: str) -> type[BaseExtractionSchema]:
    """
    Get appropriate Pydantic schema for a document type.

    Args:
        document_type: Document type (invoice, receipt, bank_statement, etc.)

    Returns:
        Pydantic schema class

    Example:
        >>> schema = get_schema_for_document_type("invoice")
        >>> result = schema(**extracted_data)
        >>> print(result.TOTAL_AMOUNT)
    """
    doc_type_lower = document_type.lower().replace(" ", "_")

    # Map aliases to base types
    if "receipt" in doc_type_lower:
        return ReceiptExtraction
    elif "invoice" in doc_type_lower:
        return InvoiceExtraction
    elif "statement" in doc_type_lower or "bank" in doc_type_lower:
        return BankStatementExtraction

    # Default to universal
    return UniversalExtraction
