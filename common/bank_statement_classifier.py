"""
Bank Statement Structure Classifier

Classifies bank statement layouts to enable structure-specific extraction.
Supports 10+ different bank statement formats including table and mobile app layouts.

Usage:
    >>> classifier = BankStatementClassifier(llm=vision_model)
    >>> structure = classifier.classify(image_path="statement.png")
    >>> print(f"Detected: {structure.structure_type}")
    >>> print(f"Confidence: {structure.confidence}")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class BankStructureResult:
    """Result from bank statement structure classification."""

    structure_type: str
    column_count: Optional[int]
    confidence: str
    reasoning: str
    extraction_guidance: Dict[str, Any]
    raw_response: str

    def is_table_format(self) -> bool:
        """Check if this is a traditional table format."""
        return self.structure_type.startswith("TABLE_")

    def is_mobile_format(self) -> bool:
        """Check if this is a mobile app format."""
        return "MOBILE_APP" in self.structure_type

    def requires_multi_turn(self) -> bool:
        """Check if this structure requires multi-turn extraction."""
        return self.extraction_guidance.get("approach") == "multi_turn_recommended"

    def get_extraction_approach(self) -> str:
        """Get recommended extraction approach."""
        return self.extraction_guidance.get("approach", "single_pass")


class BankStatementClassifier:
    """
    Classifies bank statement structural layouts for optimized extraction.

    Supports:
    - 6 table formats (3-col, 4-col, 5-col, multi-column)
    - 3 mobile app formats (dark, light inline, light summary)
    - Date-grouped formats
    - Model-specific classification prompts (Llama vs InternVL3)

    Example:
        >>> from langchain_llm import get_vision_llm
        >>> llm = get_vision_llm("llama-3.2-vision")
        >>> classifier = BankStatementClassifier(llm=llm, model_name="llama-3.2-vision")
        >>>
        >>> # Classify a bank statement
        >>> result = classifier.classify(image_path="statement.png")
        >>> print(f"Structure: {result.structure_type}")
        >>> print(f"Approach: {result.get_extraction_approach()}")
        >>>
        >>> # Get extraction guidance
        >>> if result.requires_multi_turn():
        ...     print("Recommend multi-turn extraction")
        >>> else:
        ...     print("Single-pass extraction OK")
    """

    def __init__(
        self,
        llm: Any,
        config_path: Optional[Path] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize bank statement classifier.

        Args:
            llm: Vision-language model (LangChain BaseChatModel)
            config_path: Path to bank_structure.yaml (uses default if None)
            model_name: Model name for model-specific prompts
        """
        self.llm = llm
        self.model_name = model_name

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "classifiers" / "bank_structure.yaml"

        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        """Load classifier configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Bank structure classifier config not found: {self.config_path}"
            )

        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Extract key sections
        self.categories = self.config.get("categories", {})
        self.extraction_guidance = self.config.get("extraction_guidance", {})
        self.model_specific_prompts = self.config.get("model_specific_prompts", {})

    def classify(
        self,
        image_path: Optional[Path] = None,
        image_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> BankStructureResult:
        """
        Classify bank statement structure.

        Args:
            image_path: Path to bank statement image
            image_url: URL to bank statement image
            model_name: Override model name for this classification

        Returns:
            BankStructureResult with classification details

        Example:
            >>> result = classifier.classify(image_path="statement.png")
            >>> if result.confidence == "HIGH":
            ...     print(f"Confidently classified as: {result.structure_type}")
        """
        if image_path is None and image_url is None:
            raise ValueError("Must provide either image_path or image_url")

        # Get classification prompt (model-aware)
        effective_model = model_name or self.model_name
        classification_prompt = self._get_classification_prompt(effective_model)

        # Build messages for vision LLM
        messages = self._build_classification_messages(
            classification_prompt,
            image_path,
            image_url
        )

        # Invoke LLM
        response = self.llm.invoke(messages)

        # Parse response
        result = self._parse_classification_result(response.content)

        return result

    def _get_classification_prompt(self, model_name: Optional[str]) -> str:
        """
        Get classification prompt for the specified model.

        Args:
            model_name: Model name (e.g., 'llama-3.2-vision', 'internvl3')

        Returns:
            Classification prompt string
        """
        # Try model-specific prompt first
        if model_name:
            # Normalize model name (e.g., llama-3.2-11b-vision → llama-3.2-vision)
            normalized_model = self._normalize_model_name(model_name)

            if normalized_model in self.model_specific_prompts:
                return self.model_specific_prompts[normalized_model].get("prompt", "")

        # Fall back to default classification prompt
        return self.config.get("classification_prompt", "")

    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize model name for prompt selection.

        Args:
            model_name: Original model name

        Returns:
            Normalized model name
        """
        if not model_name:
            return ""

        # Llama family
        if "llama" in model_name.lower() and "3.2" in model_name:
            return "llama-3.2-vision"

        # InternVL3 family
        if "internvl3" in model_name.lower() or "intern-vl3" in model_name.lower():
            return "internvl3"

        return model_name

    def _build_classification_messages(
        self,
        prompt: str,
        image_path: Optional[Path],
        image_url: Optional[str],
    ) -> list:
        """
        Build messages for vision LLM.

        Args:
            prompt: Classification prompt text
            image_path: Path to image
            image_url: URL to image

        Returns:
            List of messages for LLM
        """
        # System message
        system_msg = SystemMessage(
            content="You are a bank statement structure classifier. Analyze the layout accurately."
        )

        # Human message with image
        if image_path:
            # Load image for vision model
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

    def _parse_classification_result(self, response_text: str) -> BankStructureResult:
        """
        Parse LLM classification response.

        Args:
            response_text: Raw LLM output

        Returns:
            Structured BankStructureResult

        Example response:
            STRUCTURE_TYPE: TABLE_5COL_STANDARD
            COLUMN_COUNT: 5
            CONFIDENCE: HIGH
            REASONING: The statement has 5 clear column headers...
        """
        lines = response_text.strip().split("\n")

        # Parse fields
        structure_type = None
        column_count = None
        confidence = None
        reasoning = ""

        for line in lines:
            line = line.strip()

            if line.startswith("STRUCTURE_TYPE:"):
                structure_type = line.split(":", 1)[1].strip()
            elif line.startswith("COLUMN_COUNT:"):
                count_str = line.split(":", 1)[1].strip()
                if count_str.lower() != "n/a" and count_str.isdigit():
                    column_count = int(count_str)
            elif line.startswith("CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Validate structure_type
        if structure_type not in self.categories:
            # Try to find best match
            structure_type = self._find_best_match(structure_type or "")

        # Get extraction guidance for this structure
        guidance = self.extraction_guidance.get(structure_type, {})

        return BankStructureResult(
            structure_type=structure_type or "UNKNOWN",
            column_count=column_count,
            confidence=confidence or "MEDIUM",
            reasoning=reasoning,
            extraction_guidance=guidance,
            raw_response=response_text,
        )

    def _find_best_match(self, partial_type: str) -> str:
        """
        Find best matching category for partial/unclear classification.

        Args:
            partial_type: Partial or unclear structure type

        Returns:
            Best matching category name
        """
        partial_lower = partial_type.lower()

        # Keyword matching
        for category in self.categories:
            if category.lower() in partial_lower or partial_lower in category.lower():
                return category

        # Default fallback
        return "TABLE_5COL_STANDARD"  # Most common format

    def get_category_info(self, structure_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a structure category.

        Args:
            structure_type: Category name

        Returns:
            Dictionary with category details

        Example:
            >>> info = classifier.get_category_info("TABLE_5COL_STANDARD")
            >>> print(info['description'])
            '5-column table: Date | Transaction | Debit | Credit | Balance'
        """
        return self.categories.get(structure_type, {})

    def list_categories(self) -> list[str]:
        """
        Get list of all supported structure categories.

        Returns:
            List of category names

        Example:
            >>> categories = classifier.list_categories()
            >>> print(f"Supports {len(categories)} formats")
        """
        return list(self.categories.keys())

    def reload_config(self):
        """
        Reload configuration from YAML file.

        Useful for hot-reloading classifier settings.

        Example:
            >>> # Edit config/classifiers/bank_structure.yaml
            >>> classifier.reload_config()
            >>> # New categories/prompts now active
        """
        self._load_config()
        print(f"✅ Classifier configuration reloaded from {self.config_path}")


# ============================================================================
# Convenience Functions
# ============================================================================

def classify_bank_statement(
    image_path: Path,
    llm: Any,
    model_name: Optional[str] = None,
) -> BankStructureResult:
    """
    Quick function to classify a bank statement structure.

    Args:
        image_path: Path to bank statement image
        llm: Vision-language model
        model_name: Model name for model-specific prompts

    Returns:
        BankStructureResult with classification

    Example:
        >>> from langchain_llm import get_vision_llm
        >>> llm = get_vision_llm("llama-3.2-vision")
        >>> result = classify_bank_statement(
        ...     image_path=Path("statement.png"),
        ...     llm=llm,
        ...     model_name="llama-3.2-vision"
        ... )
        >>> print(f"Detected: {result.structure_type}")
    """
    classifier = BankStatementClassifier(llm=llm, model_name=model_name)
    return classifier.classify(image_path=image_path)
