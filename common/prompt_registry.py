"""
Prompt Registry - Centralized Prompt Management System

Provides unified access to all prompt variants with version control and A/B testing support.

Usage:
    >>> registry = PromptRegistry()
    >>> prompt = registry.get_prompt(
    ...     document_type="invoice",
    ...     model_name="llama-3.2-vision",
    ...     version="2.0"
    ... )
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from .bank_statement_classifier import BankStatementClassifier
from .langchain_prompts import LangChainPromptManager
from .multiturn_extractor import MultiTurnExtractor


@dataclass
class PromptMetadata:
    """Metadata for a prompt variant."""

    name: str
    version: str
    model_compatibility: List[str]
    accuracy_baseline: Optional[float] = None
    description: str = ""
    active: bool = True


class PromptRegistry:
    """
    Central registry for all prompt variants.

    Manages:
    - Model-specific prompts (Llama vs InternVL3)
    - Document-type-specific prompts
    - Structure-specific prompts (bank statements)
    - Prompt versioning and A/B testing
    - Multi-turn extraction workflows

    Example:
        >>> registry = PromptRegistry()
        >>>
        >>> # Get optimized prompt for model + document type
        >>> prompt = registry.get_prompt(
        ...     document_type="invoice",
        ...     model_name="llama-3.2-vision"
        ... )
        >>>
        >>> # Get bank statement classifier
        >>> classifier = registry.get_bank_classifier(llm=vision_model)
        >>>
        >>> # Get multi-turn extractor
        >>> extractor = registry.get_multiturn_extractor(llm=vision_model)
    """

    def __init__(self):
        """Initialize prompt registry."""
        self._prompt_managers: Dict[str, LangChainPromptManager] = {}
        self._classifiers: Dict[str, BankStatementClassifier] = {}
        self._extractors: Dict[str, MultiTurnExtractor] = {}

    def get_prompt(
        self,
        document_type: str,
        model_name: Optional[str] = None,
        structure_type: Optional[str] = None,
        version: Optional[str] = None,
    ) -> ChatPromptTemplate:
        """
        Get prompt from registry.

        Args:
            document_type: Type of document (invoice, receipt, bank_statement)
            model_name: Model name for model-specific prompts
            structure_type: Bank statement structure (if applicable)
            version: Prompt version (defaults to latest)

        Returns:
            ChatPromptTemplate configured for these parameters

        Example:
            >>> # Get Llama-optimized invoice prompt
            >>> prompt = registry.get_prompt(
            ...     document_type="invoice",
            ...     model_name="llama-3.2-vision"
            ... )
            >>>
            >>> # Get structure-specific bank statement prompt
            >>> prompt = registry.get_prompt(
            ...     document_type="bank_statement",
            ...     model_name="llama-3.2-vision",
            ...     structure_type="TABLE_5COL_STANDARD"
            ... )
        """
        # Get or create prompt manager for this model
        manager = self._get_or_create_manager(model_name)

        # Get base prompt
        prompt = manager.get_extraction_prompt(
            document_type=document_type,
            model_name=model_name,
        )

        # TODO: Apply structure-specific modifications if needed
        # TODO: Apply version-specific modifications if needed

        return prompt

    def get_bank_classifier(
        self,
        llm: Any,
        model_name: Optional[str] = None,
    ) -> BankStatementClassifier:
        """
        Get bank statement classifier.

        Args:
            llm: Vision-language model
            model_name: Model name for model-specific classification prompts

        Returns:
            BankStatementClassifier instance

        Example:
            >>> classifier = registry.get_bank_classifier(
            ...     llm=vision_model,
            ...     model_name="llama-3.2-vision"
            ... )
            >>> result = classifier.classify(image_path="statement.png")
        """
        cache_key = f"{model_name}_{id(llm)}"

        if cache_key not in self._classifiers:
            self._classifiers[cache_key] = BankStatementClassifier(
                llm=llm,
                model_name=model_name,
            )

        return self._classifiers[cache_key]

    def get_multiturn_extractor(self, llm: Any) -> MultiTurnExtractor:
        """
        Get multi-turn extractor.

        Args:
            llm: Vision-language model

        Returns:
            MultiTurnExtractor instance

        Example:
            >>> extractor = registry.get_multiturn_extractor(llm=vision_model)
            >>> result = extractor.extract_bank_statement(image_path="complex.png")
        """
        cache_key = id(llm)

        if cache_key not in self._extractors:
            self._extractors[cache_key] = MultiTurnExtractor(llm=llm)

        return self._extractors[cache_key]

    def _get_or_create_manager(
        self,
        model_name: Optional[str],
    ) -> LangChainPromptManager:
        """Get or create prompt manager for model."""
        cache_key = model_name or "default"

        if cache_key not in self._prompt_managers:
            self._prompt_managers[cache_key] = LangChainPromptManager(
                model_name=model_name
            )

        return self._prompt_managers[cache_key]

    def list_models(self) -> List[str]:
        """
        List all supported model names.

        Returns:
            List of model names with specific prompts

        Example:
            >>> models = registry.list_models()
            >>> print(f"Supports {len(models)} models")
        """
        manager = self._get_or_create_manager(None)

        if hasattr(manager, '_yaml_config'):
            model_adaptations = manager._yaml_config.config.get('model_adaptations', {})
            return list(model_adaptations.keys())

        return []

    def list_document_types(self) -> List[str]:
        """
        List all supported document types.

        Returns:
            List of document types

        Example:
            >>> doc_types = registry.list_document_types()
            >>> print(doc_types)
            ['invoice', 'receipt', 'bank_statement']
        """
        return ["invoice", "receipt", "bank_statement"]

    def get_model_info(self, model_name: str) -> Dict:
        """
        Get information about a model's characteristics.

        Args:
            model_name: Model name

        Returns:
            Dictionary with model characteristics

        Example:
            >>> info = registry.get_model_info("llama-3.2-vision")
            >>> print(f"Style: {info['style']}")
            >>> print(f"Verbosity: {info['verbosity']}")
        """
        manager = self._get_or_create_manager(model_name)
        return manager.get_model_info(model_name)

    def reload_all(self):
        """
        Reload all configurations.

        Useful for hot-reloading after editing YAML files.

        Example:
            >>> # Edit config/prompts.yaml
            >>> registry.reload_all()
            >>> # All prompts now use updated configuration
        """
        for manager in self._prompt_managers.values():
            manager.reload_config()

        for classifier in self._classifiers.values():
            classifier.reload_config()

        print("✅ All registry configurations reloaded")


# ============================================================================
# Global Registry Instance
# ============================================================================

# Singleton registry for easy access
_global_registry: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    """
    Get global prompt registry instance.

    Returns:
        Global PromptRegistry singleton

    Example:
        >>> from common.prompt_registry import get_registry
        >>>
        >>> registry = get_registry()
        >>> prompt = registry.get_prompt("invoice", "llama-3.2-vision")
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = PromptRegistry()

    return _global_registry
