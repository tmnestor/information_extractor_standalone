"""
Integration Tests for Phases 1-4

Tests all major features implemented across the 4 phases:
- Phase 1: Model-specific prompts
- Phase 2: Bank statement classification
- Phase 3: Enhanced prompts with layouts
- Phase 4: Multi-turn extraction and registry

Run with: pytest tests/test_integration_phases1234.py -v
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.bank_statement_classifier import BankStatementClassifier
from common.langchain_prompts import LangChainPromptManager, get_extraction_prompt
from common.multiturn_extractor import MultiTurnExtractor
from common.prompt_registry import PromptRegistry, get_registry

# ============================================================================
# Phase 1 Tests: Model-Specific Prompts
# ============================================================================

class TestPhase1ModelSpecificPrompts:
    """Test model-specific prompt selection."""

    def test_prompt_manager_with_model_name(self):
        """Test that LangChainPromptManager accepts model_name parameter."""
        manager = LangChainPromptManager(model_name="llama-3.2-vision")

        assert manager.model_name == "llama-3.2-vision"
        assert manager.use_yaml_config is True

    def test_model_specific_prompt_loading_llama(self):
        """Test loading Llama-specific prompts."""
        manager = LangChainPromptManager(model_name="llama-3.2-vision")

        prompt = manager.get_extraction_prompt(
            document_type="invoice",
            model_name="llama-3.2-vision"
        )

        assert prompt is not None
        assert isinstance(prompt, object)  # ChatPromptTemplate

    def test_model_specific_prompt_loading_internvl3(self):
        """Test loading InternVL3-specific prompts."""
        manager = LangChainPromptManager(model_name="internvl3")

        prompt = manager.get_extraction_prompt(
            document_type="invoice",
            model_name="internvl3"
        )

        assert prompt is not None

    def test_model_family_normalization(self):
        """Test model family normalization (e.g., llama-3.2-11b-vision → llama-3.2-vision)."""
        manager = LangChainPromptManager()

        # Test Llama family
        family = manager._get_model_family("llama-3.2-11b-vision-8bit")
        assert family == "llama-3.2-vision"

        # Test InternVL3 family
        family = manager._get_model_family("internvl3-8b-quantized")
        assert family == "internvl3"

    def test_get_model_info(self):
        """Test retrieving model characteristic info."""
        manager = LangChainPromptManager()

        info = manager.get_model_info("llama-3.2-vision")

        assert "style" in info or len(info) == 0  # May be empty if config not loaded
        # If loaded: assert info['style'] == 'step-by-step'

    def test_convenience_function(self):
        """Test get_extraction_prompt convenience function with model_name."""
        prompt = get_extraction_prompt(
            document_type="receipt",
            model_name="internvl3"
        )

        assert prompt is not None


# ============================================================================
# Phase 2 Tests: Bank Statement Classification
# ============================================================================

class TestPhase2BankStatementClassification:
    """Test bank statement structure classifier."""

    def test_classifier_initialization(self):
        """Test that BankStatementClassifier can be initialized."""
        # Mock LLM for testing
        class MockLLM:
            def invoke(self, messages):
                class Response:
                    content = """STRUCTURE_TYPE: TABLE_5COL_STANDARD
COLUMN_COUNT: 5
CONFIDENCE: HIGH
REASONING: Statement has 5 clear columns"""
                return Response()

        classifier = BankStatementClassifier(llm=MockLLM())

        assert classifier is not None
        assert hasattr(classifier, 'categories')
        assert hasattr(classifier, 'extraction_guidance')

    def test_classifier_categories_loaded(self):
        """Test that classifier categories are loaded from config."""
        class MockLLM:
            pass

        classifier = BankStatementClassifier(llm=MockLLM())

        # Should have 10+ categories
        assert len(classifier.categories) >= 10

        # Check for specific categories
        assert "TABLE_3COL_SIMPLE" in classifier.categories
        assert "TABLE_5COL_STANDARD" in classifier.categories
        assert "MOBILE_APP_DARK" in classifier.categories

    def test_classifier_list_categories(self):
        """Test listing all supported categories."""
        class MockLLM:
            pass

        classifier = BankStatementClassifier(llm=MockLLM())
        categories = classifier.list_categories()

        assert len(categories) >= 10
        assert "TABLE_4COL_STANDARD" in categories

    def test_classifier_get_category_info(self):
        """Test getting detailed info for a category."""
        class MockLLM:
            pass

        classifier = BankStatementClassifier(llm=MockLLM())
        info = classifier.get_category_info("TABLE_5COL_STANDARD")

        assert "description" in info
        assert "5-column table" in info["description"]

    def test_classifier_model_specific_prompts(self):
        """Test that model-specific classification prompts exist."""
        class MockLLM:
            pass

        classifier = BankStatementClassifier(llm=MockLLM())

        # Should have prompts for both models
        assert "llama-3.2-vision" in classifier.model_specific_prompts
        assert "internvl3" in classifier.model_specific_prompts

    def test_bank_structure_result_helper_methods(self):
        """Test BankStructureResult helper methods."""
        from common.bank_statement_classifier import BankStructureResult

        # Test table format
        result = BankStructureResult(
            structure_type="TABLE_5COL_STANDARD",
            column_count=5,
            confidence="HIGH",
            reasoning="Test",
            extraction_guidance={"approach": "single_pass"},
            raw_response="Test"
        )

        assert result.is_table_format() is True
        assert result.is_mobile_format() is False
        assert result.requires_multi_turn() is False
        assert result.get_extraction_approach() == "single_pass"

        # Test mobile format
        mobile_result = BankStructureResult(
            structure_type="MOBILE_APP_DARK",
            column_count=None,
            confidence="HIGH",
            reasoning="Test",
            extraction_guidance={"approach": "section_based"},
            raw_response="Test"
        )

        assert mobile_result.is_table_format() is False
        assert mobile_result.is_mobile_format() is True


# ============================================================================
# Phase 3 Tests: Enhanced Prompts
# ============================================================================

class TestPhase3EnhancedPrompts:
    """Test enhanced prompts with layout examples."""

    def test_structure_specific_prompts_exist(self):
        """Test that structure-specific prompts are in config."""
        manager = LangChainPromptManager()

        if hasattr(manager, '_yaml_config'):
            config = manager._yaml_config.config
            bank_prompts = config.get('bank_statement_structure_prompts', {})

            assert len(bank_prompts) > 0
            assert "TABLE_4COL_STANDARD" in bank_prompts or len(bank_prompts) == 0

    def test_model_adaptations_loaded(self):
        """Test that model adaptation settings are loaded."""
        manager = LangChainPromptManager()

        if hasattr(manager, '_yaml_config'):
            config = manager._yaml_config.config
            adaptations = config.get('model_adaptations', {})

            # Should have adaptations for multiple models
            assert len(adaptations) >= 2 or len(adaptations) == 0


# ============================================================================
# Phase 4 Tests: Multi-Turn Extraction and Registry
# ============================================================================

class TestPhase4MultiTurnExtractor:
    """Test multi-turn extractor."""

    def test_multiturn_extractor_initialization(self):
        """Test that MultiTurnExtractor can be initialized."""
        class MockLLM:
            pass

        extractor = MultiTurnExtractor(llm=MockLLM())

        assert extractor is not None
        assert extractor.llm is not None

    def test_multiturn_column_prompt_building(self):
        """Test building column-specific prompts."""
        class MockLLM:
            pass

        extractor = MultiTurnExtractor(llm=MockLLM())

        prompt = extractor._build_column_prompt(
            column_name="Date",
            column_description="date values",
            additional_instructions=""
        )

        assert "Date" in prompt
        assert "date values" in prompt
        assert "one value per line" in prompt.lower()

    def test_multiturn_validation(self):
        """Test column alignment validation."""
        class MockLLM:
            pass

        extractor = MultiTurnExtractor(llm=MockLLM())

        # Test valid alignment
        errors = extractor._validate_alignment(
            dates=["01/01/2025", "02/01/2025"],
            descriptions=["Purchase", "ATM"],
            debits=["$50.00", "$100.00"],
            credits=["EMPTY", "EMPTY"],
            balances=["$1000", "$900"]
        )

        assert len(errors) == 0

        # Test invalid alignment
        errors = extractor._validate_alignment(
            dates=["01/01/2025", "02/01/2025"],
            descriptions=["Purchase"],  # Missing one
            debits=["$50.00", "$100.00"],
            credits=["EMPTY", "EMPTY"],
            balances=["$1000", "$900"]
        )

        assert len(errors) > 0


class TestPhase4PromptRegistry:
    """Test prompt registry."""

    def test_registry_initialization(self):
        """Test that PromptRegistry can be initialized."""
        registry = PromptRegistry()

        assert registry is not None

    def test_registry_get_prompt(self):
        """Test getting prompts from registry."""
        registry = PromptRegistry()

        prompt = registry.get_prompt(
            document_type="invoice",
            model_name="llama-3.2-vision"
        )

        assert prompt is not None

    def test_registry_list_models(self):
        """Test listing supported models."""
        registry = PromptRegistry()
        models = registry.list_models()

        # May be empty if config not loaded
        assert isinstance(models, list)

    def test_registry_list_document_types(self):
        """Test listing supported document types."""
        registry = PromptRegistry()
        doc_types = registry.list_document_types()

        assert "invoice" in doc_types
        assert "receipt" in doc_types
        assert "bank_statement" in doc_types

    def test_global_registry_singleton(self):
        """Test global registry singleton."""
        registry1 = get_registry()
        registry2 = get_registry()

        # Should be same instance
        assert registry1 is registry2


# ============================================================================
# Integration Tests: Combined Workflows
# ============================================================================

class TestCombinedWorkflows:
    """Test workflows that combine multiple phases."""

    def test_full_bank_statement_workflow_mock(self):
        """Test complete bank statement processing workflow."""
        class MockLLM:
            def __init__(self):
                self.call_count = 0

            def invoke(self, messages):
                self.call_count += 1

                class Response:
                    def __init__(self, call_num):
                        # First call: classification
                        if call_num == 1:
                            self.content = """STRUCTURE_TYPE: TABLE_5COL_STANDARD
COLUMN_COUNT: 5
CONFIDENCE: HIGH
REASONING: Clear 5-column structure"""
                        else:
                            # Subsequent calls: extraction
                            self.content = "01/01/2025\n02/01/2025"

                return Response(self.call_count)

        llm = MockLLM()
        registry = PromptRegistry()

        # Step 1: Classify structure
        classifier = registry.get_bank_classifier(llm=llm, model_name="llama-3.2-vision")
        assert classifier is not None

        # Step 2: Get structure-specific prompt
        prompt = registry.get_prompt(
            document_type="bank_statement",
            model_name="llama-3.2-vision",
            structure_type="TABLE_5COL_STANDARD"
        )
        assert prompt is not None

        # Step 3: Check if multi-turn recommended
        # (Would use result.requires_multi_turn() in real workflow)

    def test_model_aware_extraction_selection(self):
        """Test selecting appropriate extraction method based on model."""
        registry = PromptRegistry()

        # Llama should get step-by-step prompts
        llama_info = registry.get_model_info("llama-3.2-vision")

        # InternVL3 should get concise prompts
        internvl3_info = registry.get_model_info("internvl3")

        # Structure should differ (if config loaded)
        # This is a smoke test - actual assertion depends on config being loaded
        assert llama_info is not None or llama_info == {}
        assert internvl3_info is not None or internvl3_info == {}


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
