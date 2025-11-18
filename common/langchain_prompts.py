"""
LangChain Prompt Management for Document Extraction

Hybrid approach combining YAML-based configuration with dynamic LangChain templates.
Enables dynamic field injection, prompt composition, and hot-reload capability.

Key Features:
- YAML-based prompt configuration (hot-reload)
- Dynamic field injection from schema
- Document-type-specific prompts
- Vision-language model support
- Prompt composition and reuse
- Backward compatibility with legacy YAML files
"""

from pathlib import Path
from typing import Dict, Optional

import yaml
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from .field_definitions_loader import SimpleFieldLoader


class LangChainPromptManager:
    """
    Manages LangChain prompts for document extraction.

    Provides dynamic prompt generation with field injection and
    document-type-specific customization.

    Usage:
        >>> manager = LangChainPromptManager()
        >>> prompt = manager.get_extraction_prompt("invoice")
        >>> messages = prompt.format_messages(image="<image>")
    """

    def __init__(
        self,
        field_loader: Optional[SimpleFieldLoader] = None,
        use_yaml_config: bool = True,
        system_mode: str = "expert",
    ):
        """
        Initialize prompt manager.

        Args:
            field_loader: Field definitions loader (creates if None)
            use_yaml_config: Load prompts from config/prompts.yaml
            system_mode: System prompt mode (expert, structured, precise, flexible, strict)
        """
        self.field_loader = field_loader or SimpleFieldLoader()
        self.use_yaml_config = use_yaml_config
        self.system_mode = system_mode

        # Cache for generated prompts
        self._prompt_cache: Dict[str, ChatPromptTemplate] = {}

        # Load configuration
        if use_yaml_config:
            try:
                # Import here to avoid circular dependency
                from .config import get_yaml_config

                self._yaml_config = get_yaml_config()

                # Load prompts from YAML
                self.system_templates = self._yaml_config.get_system_prompts()
                self.conversation_protocol = self._yaml_config.get_conversation_protocol()
                self.extraction_rules = self._yaml_config.get_extraction_rules()

            except Exception as e:
                print(f"Warning: Could not load YAML config, using defaults: {e}")
                self._load_defaults()
        else:
            self._load_defaults()

    def _load_defaults(self):
        """Load default (hard-coded) prompts as fallback."""
        # System message templates (reusable)
        self.system_templates = {
            "expert": "You are an expert document analyzer specialized in {document_type} extraction.",
            "structured": "You are a structured data extraction system. Extract information with perfect formatting.",
            "precise": "Extract only clearly visible information. Use NOT_FOUND for any unclear fields.",
        }

        # Conversation protocol (common to all prompts)
        self.conversation_protocol = """
CONVERSATION PROTOCOL:
- Start your response immediately with "DOCUMENT_TYPE:"
- Do NOT include conversational text like "I'll extract..." or "Based on the document..."
- Do NOT use bullet points, numbered lists, asterisks, or markdown formatting
- Output ONLY the structured extraction data
- End immediately after the last field with no additional text
"""

        # Extraction rules (common)
        self.extraction_rules = """
RULES:
- Use exact text from document
- CRITICAL: Use ONLY pipe separators (|) for lists - NEVER use commas
- Be conservative: use NOT_FOUND if field is truly missing
- Stop after the last field - no explanations or comments
"""

    def reload_config(self):
        """
        Reload YAML configuration and clear cache.

        Enables hot-reload of prompts without restarting.

        Example:
            >>> manager = LangChainPromptManager()
            >>> # ... edit config/prompts.yaml ...
            >>> manager.reload_config()  # Pick up changes
        """
        if self.use_yaml_config and hasattr(self, '_yaml_config'):
            self._yaml_config.reload()
            # Reload prompts from YAML
            self.system_templates = self._yaml_config.get_system_prompts()
            self.conversation_protocol = self._yaml_config.get_conversation_protocol()
            self.extraction_rules = self._yaml_config.get_extraction_rules()

        # Clear cache to force regeneration
        self._prompt_cache.clear()

        print("✅ Prompt configuration reloaded")

    def get_extraction_prompt(
        self,
        document_type: str,
        include_format_instructions: bool = True,
    ) -> ChatPromptTemplate:
        """
        Get extraction prompt for document type.

        Args:
            document_type: Type of document (invoice, receipt, bank_statement)
            include_format_instructions: Include output format in prompt

        Returns:
            ChatPromptTemplate configured for this document type

        Example:
            >>> prompt = manager.get_extraction_prompt("invoice")
            >>> messages = prompt.format_messages()
        """
        cache_key = f"{document_type}_{include_format_instructions}"

        if cache_key not in self._prompt_cache:
            self._prompt_cache[cache_key] = self._build_extraction_prompt(
                document_type, include_format_instructions
            )

        return self._prompt_cache[cache_key]

    def get_detection_prompt(self) -> ChatPromptTemplate:
        """
        Get document type detection prompt.

        Returns:
            ChatPromptTemplate for document type detection
        """
        if "detection" not in self._prompt_cache:
            self._prompt_cache["detection"] = self._build_detection_prompt()

        return self._prompt_cache["detection"]

    def _build_extraction_prompt(
        self,
        document_type: str,
        include_format_instructions: bool,
    ) -> ChatPromptTemplate:
        """
        Build extraction prompt from field definitions.

        Args:
            document_type: Type of document
            include_format_instructions: Include format section

        Returns:
            ChatPromptTemplate instance
        """
        # Get fields for this document type
        fields = self.field_loader.get_document_fields(document_type)
        field_count = len(fields)

        # Build field format section
        field_lines = []
        for field in fields:
            description = self.field_loader.get_field_description(field)
            field_lines.append(f"{field}: [{description}]")

        field_format = "\n".join(field_lines)

        # Get document-specific instructions
        doc_instructions = self._get_document_instructions(document_type)

        # Build prompt components
        # Use configured system mode
        system_template = self.system_templates.get(
            self.system_mode,
            self.system_templates.get("expert", "You are an expert document analyzer.")
        )
        system_message = SystemMessagePromptTemplate.from_template(system_template)

        # Human message with image + text
        extraction_template = f"""Extract structured data from this business document image.

{self.conversation_protocol}

{doc_instructions}

OUTPUT FORMAT ({field_count} FIELDS):

{field_format}

{self.extraction_rules}"""

        # For vision models, we need to include image placeholder
        human_message = HumanMessagePromptTemplate.from_template(
            [
                {"type": "image"},  # Placeholder for image
                {"type": "text", "text": extraction_template}
            ]
        )

        # Combine into chat prompt
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            human_message
        ])

        # Pre-fill document_type variable
        prompt = prompt.partial(document_type=document_type)

        return prompt

    def _build_detection_prompt(self) -> ChatPromptTemplate:
        """
        Build document type detection prompt.

        Returns:
            ChatPromptTemplate for detection
        """
        system_message = SystemMessagePromptTemplate.from_template(
            "You are a document classification system. Identify document types accurately."
        )

        detection_text = """Identify the document type from this image.

VALID TYPES:
- INVOICE (tax invoice, invoice, bill)
- RECEIPT (purchase receipt, sales receipt)
- BANK_STATEMENT (bank statement, account statement)

OUTPUT FORMAT:
DOCUMENT_TYPE: [one of the valid types above]

Respond with ONLY the document type line - no explanations."""

        human_message = HumanMessagePromptTemplate.from_template(
            [
                {"type": "image"},
                {"type": "text", "text": detection_text}
            ]
        )

        return ChatPromptTemplate.from_messages([system_message, human_message])

    def _get_document_instructions(self, document_type: str) -> str:
        """
        Get document-type-specific instructions.

        Args:
            document_type: Type of document

        Returns:
            Additional instructions for this document type
        """
        # Try to load from YAML config first
        if self.use_yaml_config and hasattr(self, '_yaml_config'):
            try:
                instructions = self._yaml_config.get_document_instructions(document_type)
                if instructions:
                    return instructions
            except Exception:
                pass  # Fall back to defaults

        # Default instructions (fallback)
        instructions = {
            "invoice": """CRITICAL INSTRUCTIONS:
- Extract ALL line items (products/services purchased)
- Calculate line totals: quantity × unit price
- Identify GST amount and total clearly
- Use pipe (|) separator for multiple items""",

            "receipt": """CRITICAL INSTRUCTIONS:
- Extract ALL purchased items from receipt
- Include quantities and individual prices
- Note if GST is included in total
- Use pipe (|) separator for multiple items""",

            "bank_statement": """CRITICAL INSTRUCTIONS:
- Extract ALL transactions from the statement
- Each transaction needs: date, description, amount
- Use pipe (|) separators between transactions
- Maintain transaction order as shown""",
        }

        return instructions.get(document_type, "")

    def load_yaml_prompt(self, yaml_file: Path, prompt_key: str) -> str:
        """
        Load prompt from YAML file (fallback compatibility).

        Args:
            yaml_file: Path to YAML prompt file
            prompt_key: Key within YAML file

        Returns:
            Prompt string

        Example:
            >>> prompt_text = manager.load_yaml_prompt(
            ...     Path("prompts/llama_prompts.yaml"),
            ...     "invoice"
            ... )
        """
        if not yaml_file.exists():
            raise FileNotFoundError(f"YAML prompt file not found: {yaml_file}")

        with yaml_file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        prompts = data.get("prompts", {})
        if prompt_key not in prompts:
            raise KeyError(f"Prompt key '{prompt_key}' not found in {yaml_file}")

        return prompts[prompt_key].get("prompt", "")

    def create_hybrid_prompt(
        self,
        yaml_file: Path,
        prompt_key: str,
    ) -> ChatPromptTemplate:
        """
        Create ChatPromptTemplate from existing YAML prompt.

        Allows gradual migration from YAML to LangChain prompts.

        Args:
            yaml_file: Path to YAML prompt file
            prompt_key: Key within YAML file

        Returns:
            ChatPromptTemplate wrapping the YAML prompt

        Example:
            >>> prompt = manager.create_hybrid_prompt(
            ...     Path("prompts/generated/llama_invoice_prompt.yaml"),
            ...     "invoice"
            ... )
        """
        # Load YAML prompt text
        prompt_text = self.load_yaml_prompt(yaml_file, prompt_key)

        # Wrap in ChatPromptTemplate
        system_message = SystemMessagePromptTemplate.from_template(
            "You are an expert document analyzer."
        )

        human_message = HumanMessagePromptTemplate.from_template(
            [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        )

        return ChatPromptTemplate.from_messages([system_message, human_message])


class PromptSelector:
    """
    Selects appropriate prompt based on document type and configuration.

    Supports:
    - LangChain-generated prompts (new)
    - YAML prompts (legacy compatibility)
    - Hybrid mode

    Usage:
        >>> selector = PromptSelector(prompt_manager)
        >>> prompt = selector.select_prompt(
        ...     document_type="invoice",
        ...     use_langchain=True
        ... )
    """

    def __init__(
        self,
        prompt_manager: LangChainPromptManager,
        yaml_prompt_base: Optional[Path] = None,
    ):
        """
        Initialize prompt selector.

        Args:
            prompt_manager: LangChain prompt manager
            yaml_prompt_base: Base directory for YAML prompts
        """
        self.prompt_manager = prompt_manager
        self.yaml_prompt_base = yaml_prompt_base or Path("prompts/generated")

    def select_prompt(
        self,
        document_type: str,
        use_langchain: bool = True,
        yaml_file: Optional[Path] = None,
    ) -> ChatPromptTemplate:
        """
        Select appropriate prompt for document type.

        Args:
            document_type: Type of document
            use_langchain: Use LangChain-generated prompts (vs YAML)
            yaml_file: Override YAML file path

        Returns:
            ChatPromptTemplate instance

        Example:
            >>> # Use new LangChain prompts
            >>> prompt = selector.select_prompt("invoice", use_langchain=True)
            >>>
            >>> # Use legacy YAML prompts
            >>> prompt = selector.select_prompt(
            ...     "invoice",
            ...     use_langchain=False,
            ...     yaml_file=Path("prompts/generated/llama_invoice_prompt.yaml")
            ... )
        """
        if use_langchain:
            # Use dynamically generated LangChain prompts
            return self.prompt_manager.get_extraction_prompt(document_type)
        else:
            # Use legacy YAML prompts
            if not yaml_file:
                # Construct default YAML path
                yaml_file = self.yaml_prompt_base / f"llama_{document_type}_prompt.yaml"

            return self.prompt_manager.create_hybrid_prompt(yaml_file, document_type)


# ============================================================================
# Convenience Functions
# ============================================================================

def get_extraction_prompt(
    document_type: str,
    use_langchain: bool = True,
    field_loader: Optional[SimpleFieldLoader] = None,
) -> ChatPromptTemplate:
    """
    Quick function to get extraction prompt.

    Args:
        document_type: Type of document
        use_langchain: Use LangChain prompts vs YAML
        field_loader: Custom field loader

    Returns:
        ChatPromptTemplate instance

    Example:
        >>> prompt = get_extraction_prompt("invoice")
        >>> messages = prompt.format_messages()
    """
    manager = LangChainPromptManager(field_loader=field_loader)

    if use_langchain:
        return manager.get_extraction_prompt(document_type)
    else:
        selector = PromptSelector(manager)
        return selector.select_prompt(document_type, use_langchain=False)


def get_detection_prompt() -> ChatPromptTemplate:
    """
    Quick function to get document type detection prompt.

    Returns:
        ChatPromptTemplate for detection

    Example:
        >>> prompt = get_detection_prompt()
        >>> messages = prompt.format_messages()
    """
    manager = LangChainPromptManager()
    return manager.get_detection_prompt()
