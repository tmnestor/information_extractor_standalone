"""
LangChain Chains for Document Processing Pipeline

Provides modular chains for document detection, field extraction, and
end-to-end processing. Replaces the monolithic 345-line _process_llama_image() method.

Key Features:
- DocumentDetectionChain: Identify document type
- FieldExtractionChain: Extract structured data
- DocumentProcessingPipeline: Complete end-to-end flow
- Vision-language model support (Llama, InternVL3)
- YAML-based configuration
- Callback integration for monitoring
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import HumanMessage

from .extraction_cleaner import ExtractionCleaner
from .extraction_schemas import BaseExtractionSchema
from .langchain_callbacks import DocumentProcessingCallback
from .langchain_llm import VisionLanguageModel
from .langchain_parsers import DocumentExtractionParser, DocumentTypeParser
from .langchain_prompts import LangChainPromptManager


class DocumentDetectionChain:
    """
    Chain for detecting document type from image.

    Simple chain that:
    1. Sends image + detection prompt to LLM
    2. Parses response to extract document type
    3. Returns normalized document type

    Usage:
        >>> chain = DocumentDetectionChain(llm=llm)
        >>> doc_type = chain.run(image_path="/path/to/image.png")
        >>> print(doc_type)  # "invoice"
    """

    def __init__(
        self,
        llm: VisionLanguageModel,
        prompt_manager: Optional[LangChainPromptManager] = None,
        verbose: bool = False,
    ):
        """
        Initialize detection chain.

        Args:
            llm: VisionLanguageModel instance
            prompt_manager: Prompt manager (creates if None)
            verbose: Enable verbose logging
        """
        self.llm = llm
        self.prompt_manager = prompt_manager or LangChainPromptManager()
        self.verbose = verbose

        # Get detection prompt
        self.prompt = self.prompt_manager.get_detection_prompt()

        # Parser for document type
        self.parser = DocumentTypeParser()

    def run(
        self,
        image_path: str,
        callbacks: Optional[List[Any]] = None,
    ) -> str:
        """
        Run detection on image.

        Args:
            image_path: Path to image file
            callbacks: Optional callbacks for monitoring

        Returns:
            Document type (normalized, lowercase)

        Example:
            >>> doc_type = chain.run("/path/to/invoice.png")
            >>> print(doc_type)  # "invoice"
        """
        if self.verbose:
            print(f"🔍 Detecting document type: {Path(image_path).name}")

        # Format prompt (no variables needed for detection)
        prompt_text = self.prompt.format()

        # Create multi-modal message
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_path}}
            ]
        )

        # Generate with LLM using BaseChatModel interface
        result = self.llm.invoke([message])
        response = result.content if hasattr(result, 'content') else str(result)

        if self.verbose:
            print(f"📄 Raw detection response: {response[:100]}...")

        # Parse document type
        doc_type = self.parser.parse(response)

        if self.verbose:
            print(f"✅ Detected type: {doc_type}")

        return doc_type

    def __call__(self, image_path: str, **kwargs) -> str:
        """Allow chain to be called directly."""
        return self.run(image_path, **kwargs)


class FieldExtractionChain:
    """
    Chain for extracting structured fields from document image.

    Handles:
    1. Selecting appropriate prompt for document type
    2. Sending image + extraction prompt to LLM
    3. Parsing response into Pydantic model
    4. Validating and cleaning extracted data

    Usage:
        >>> chain = FieldExtractionChain(llm=llm, document_type="invoice")
        >>> result = chain.run(image_path="/path/to/invoice.png")
        >>> print(result.TOTAL_AMOUNT)  # Decimal('123.45')
    """

    def __init__(
        self,
        llm: VisionLanguageModel,
        document_type: str = "universal",
        prompt_manager: Optional[LangChainPromptManager] = None,
        enable_fixing: bool = False,
        enable_cleaning: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize extraction chain.

        Args:
            llm: VisionLanguageModel instance
            document_type: Type of document to extract
            prompt_manager: Prompt manager (creates if None)
            enable_fixing: Enable self-healing parser
            enable_cleaning: Enable ExtractionCleaner (default: True)
            verbose: Enable verbose logging
        """
        self.llm = llm
        self.document_type = document_type
        self.prompt_manager = prompt_manager or LangChainPromptManager()
        self.enable_fixing = enable_fixing
        self.enable_cleaning = enable_cleaning
        self.verbose = verbose

        # Get extraction prompt for this document type
        self.prompt = self.prompt_manager.get_extraction_prompt(document_type)

        # Create parser for this document type
        self.parser = DocumentExtractionParser(
            document_type=document_type,
            llm=llm if enable_fixing else None,
            enable_fixing=enable_fixing,
            verbose=verbose,
        )

        # Create cleaner for field normalization (CRITICAL for accuracy)
        self.cleaner = ExtractionCleaner(debug=verbose) if enable_cleaning else None

    def run(
        self,
        image_path: str,
        callbacks: Optional[List[Any]] = None,
    ) -> BaseExtractionSchema:
        """
        Run extraction on image.

        Args:
            image_path: Path to image file
            callbacks: Optional callbacks for monitoring

        Returns:
            Pydantic model with extracted fields

        Example:
            >>> result = chain.run("/path/to/invoice.png")
            >>> print(result.SUPPLIER_NAME)
            >>> print(result.TOTAL_AMOUNT)
        """
        if self.verbose:
            print(f"📊 Extracting {self.document_type} fields: {Path(image_path).name}")

        # Format prompt
        prompt_text = self.prompt.format()

        # Create multi-modal message
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_path}}
            ]
        )

        # Generate with LLM using BaseChatModel interface
        result = self.llm.invoke([message])
        response = result.content if hasattr(result, 'content') else str(result)

        if self.verbose:
            print(f"📄 Raw extraction response: {response[:200]}...")

        # Parse into Pydantic model
        if self.enable_fixing:
            parsed_result = self.parser.parse_with_fixing(response)
        else:
            parsed_result = self.parser.parse(response)

        if self.verbose:
            # Count non-NOT_FOUND fields
            extracted_count = sum(
                1 for field, value in parsed_result.model_dump().items()
                if value != "NOT_FOUND" and value != [] and value
            )
            print(f"✅ Extracted {extracted_count} fields")

        # CRITICAL: Clean and normalize extracted values (restores 81.8% accuracy)
        if self.cleaner:
            if self.verbose:
                print("🧹 Cleaning and normalizing extracted fields...")

            # Convert Pydantic model to dict
            extracted_dict = parsed_result.model_dump()

            # Clean all fields using ExtractionCleaner
            cleaned_dict = self.cleaner.clean_extraction_dict(extracted_dict)

            # Get schema class for this document type
            from .extraction_schemas import get_schema_for_document_type
            schema_class = get_schema_for_document_type(self.document_type)

            # Create new Pydantic model from cleaned data
            parsed_result = schema_class(**cleaned_dict)

            if self.verbose:
                cleaned_count = sum(
                    1 for field, value in cleaned_dict.items()
                    if value != "NOT_FOUND" and value != [] and value
                )
                print(f"✅ Cleaned {cleaned_count} fields")

        return parsed_result

    def __call__(self, image_path: str, **kwargs) -> BaseExtractionSchema:
        """Allow chain to be called directly."""
        return self.run(image_path, **kwargs)


class DocumentProcessingPipeline:
    """
    Complete end-to-end document processing pipeline.

    Orchestrates:
    1. Document type detection
    2. Field extraction (with document-specific prompt)
    3. Validation and cleaning
    4. Metrics collection

    Replaces the 345-line _process_llama_image() method with modular chains.

    Usage:
        >>> pipeline = DocumentProcessingPipeline(llm=llm)
        >>> result = pipeline.process(image_path="/path/to/document.png")
        >>> print(result["document_type"])
        >>> print(result["extracted_data"].TOTAL_AMOUNT)
        >>> print(result["metrics"])
    """

    def __init__(
        self,
        llm: VisionLanguageModel,
        prompt_manager: Optional[LangChainPromptManager] = None,
        enable_fixing: bool = False,
        enable_cleaning: bool = True,
        skip_detection: bool = False,
        default_document_type: str = "universal",
        verbose: bool = False,
    ):
        """
        Initialize processing pipeline.

        Args:
            llm: VisionLanguageModel instance
            prompt_manager: Prompt manager (creates if None)
            enable_fixing: Enable self-healing parsers
            enable_cleaning: Enable ExtractionCleaner (default: True, critical for accuracy)
            skip_detection: Skip detection step (use default_document_type)
            default_document_type: Document type if skipping detection
            verbose: Enable verbose logging
        """
        self.llm = llm
        self.prompt_manager = prompt_manager or LangChainPromptManager()
        self.enable_fixing = enable_fixing
        self.enable_cleaning = enable_cleaning
        self.skip_detection = skip_detection
        self.default_document_type = default_document_type
        self.verbose = verbose

        # Initialize detection chain (unless skipped)
        self.detection_chain = None
        if not skip_detection:
            self.detection_chain = DocumentDetectionChain(
                llm=llm,
                prompt_manager=prompt_manager,
                verbose=verbose,
            )

        # Cache extraction chains by document type
        self._extraction_chains: Dict[str, FieldExtractionChain] = {}

    def process(
        self,
        image_path: str,
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process document through complete pipeline.

        Args:
            image_path: Path to document image
            callbacks: Optional callbacks for monitoring

        Returns:
            Dictionary containing:
                - document_type: Detected type
                - extracted_data: Pydantic model with fields
                - metrics: LLM usage metrics
                - image_path: Original image path

        Example:
            >>> result = pipeline.process("/path/to/invoice.png")
            >>> print(f"Type: {result['document_type']}")
            >>> print(f"Total: {result['extracted_data'].TOTAL_AMOUNT}")
            >>> print(f"Tokens: {result['metrics']['total_tokens_used']}")
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {Path(image_path).name}")
            print(f"{'='*60}")

        # Notify callbacks of chain start
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, "on_chain_start"):
                    callback.on_chain_start(
                        {"name": "DocumentProcessingPipeline"},
                        {"image_path": image_path},
                    )

        try:
            # Step 1: Document Type Detection
            if self.skip_detection:
                document_type = self.default_document_type
                if self.verbose:
                    print(f"⏭️  Skipping detection, using: {document_type}")
            else:
                if self.verbose:
                    print("\n📋 Step 1: Document Type Detection")
                document_type = self.detection_chain.run(image_path, callbacks)

            # Step 2: Field Extraction
            if self.verbose:
                print(f"\n📊 Step 2: Field Extraction ({document_type})")

            # Get or create extraction chain for this document type
            if document_type not in self._extraction_chains:
                self._extraction_chains[document_type] = FieldExtractionChain(
                    llm=self.llm,
                    document_type=document_type,
                    prompt_manager=self.prompt_manager,
                    enable_fixing=self.enable_fixing,
                    enable_cleaning=self.enable_cleaning,
                    verbose=self.verbose,
                )

            extraction_chain = self._extraction_chains[document_type]
            extracted_data = extraction_chain.run(image_path, callbacks)

            # Step 3: Collect Metrics
            metrics = self.llm.get_metrics()

            # Build result
            result = {
                "document_type": document_type,
                "extracted_data": extracted_data,
                "metrics": metrics,
                "image_path": image_path,
                "image_name": Path(image_path).name,
            }

            # Notify callbacks of success
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_chain_end"):
                        callback.on_chain_end(result)

                    # Document-specific callback
                    if isinstance(callback, DocumentProcessingCallback):
                        fields_extracted = sum(
                            1 for v in extracted_data.model_dump().values()
                            if v != "NOT_FOUND" and v != [] and v
                        )
                        callback.log_document_processed(
                            document_type=document_type,
                            fields_extracted=fields_extracted,
                        )

            if self.verbose:
                print("\n✅ Processing complete!")
                print(f"{'='*60}\n")

            return result

        except Exception as e:
            # Notify callbacks of error
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, "on_chain_error"):
                        callback.on_chain_error(e)

            if self.verbose:
                print(f"\n❌ Processing failed: {e}")
                print(f"{'='*60}\n")

            raise

    def process_batch(
        self,
        image_paths: List[str],
        callbacks: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images through pipeline.

        Args:
            image_paths: List of image file paths
            callbacks: Optional callbacks for monitoring

        Returns:
            List of result dictionaries (one per image)

        Example:
            >>> results = pipeline.process_batch([
            ...     "/path/to/invoice1.png",
            ...     "/path/to/invoice2.png",
            ... ])
            >>> for result in results:
            ...     print(result['document_type'], result['extracted_data'].TOTAL_AMOUNT)
        """
        if self.verbose:
            print(f"\n🔄 Processing batch of {len(image_paths)} images")

        # Notify callbacks of batch start
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, "start_batch"):
                    callback.start_batch(
                        total_items=len(image_paths),
                        description="Processing documents",
                    )

        results = []

        for i, image_path in enumerate(image_paths, 1):
            if self.verbose:
                print(f"\n[{i}/{len(image_paths)}]", end=" ")

            try:
                result = self.process(image_path, callbacks)
                results.append(result)

                # Update batch progress
                if callbacks:
                    for callback in callbacks:
                        if hasattr(callback, "update_batch"):
                            callback.update_batch(advance=1)

            except Exception as e:
                if self.verbose:
                    print(f"❌ Error processing {Path(image_path).name}: {e}")

                # Add error result
                results.append({
                    "document_type": "error",
                    "extracted_data": None,
                    "error": str(e),
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                })

        # End batch
        if callbacks:
            for callback in callbacks:
                if hasattr(callback, "end_batch"):
                    callback.end_batch()

        if self.verbose:
            success_count = sum(1 for r in results if r.get("document_type") != "error")
            print(f"\n✅ Batch complete: {success_count}/{len(image_paths)} successful")

        return results

    def __call__(self, image_path: Union[str, List[str]], **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Allow pipeline to be called directly.

        Automatically handles single image or batch.
        """
        if isinstance(image_path, list):
            return self.process_batch(image_path, **kwargs)
        else:
            return self.process(image_path, **kwargs)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_pipeline(
    llm: VisionLanguageModel,
    enable_fixing: bool = False,
    enable_cleaning: bool = True,
    verbose: bool = False,
) -> DocumentProcessingPipeline:
    """
    Quick function to create a processing pipeline.

    Args:
        llm: VisionLanguageModel instance
        enable_fixing: Enable self-healing parsers
        enable_cleaning: Enable ExtractionCleaner (default: True, critical for accuracy)
        verbose: Enable verbose logging

    Returns:
        DocumentProcessingPipeline instance

    Example:
        >>> from common.langchain_llm import VisionLanguageModelFactory
        >>> from common.llama_model_loader import load_llama_model
        >>>
        >>> # Load model with YAML config
        >>> llm = VisionLanguageModelFactory.from_yaml_config(
        ...     model_name="llama-3.2-11b-vision-8bit",
        ...     model_loader_func=load_llama_model,
        ...     verbose=True
        ... )
        >>>
        >>> # Create pipeline
        >>> pipeline = create_pipeline(llm, enable_fixing=True, enable_cleaning=True, verbose=True)
        >>>
        >>> # Process document
        >>> result = pipeline.process("/path/to/invoice.png")
    """
    return DocumentProcessingPipeline(
        llm=llm,
        enable_fixing=enable_fixing,
        enable_cleaning=enable_cleaning,
        verbose=verbose,
    )


def process_document(
    image_path: str,
    llm: VisionLanguageModel,
    callbacks: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Quick function to process a single document.

    Args:
        image_path: Path to document image
        llm: VisionLanguageModel instance
        callbacks: Optional callbacks

    Returns:
        Result dictionary

    Example:
        >>> result = process_document("/path/to/invoice.png", llm)
        >>> print(result['document_type'])
        >>> print(result['extracted_data'].TOTAL_AMOUNT)
    """
    pipeline = DocumentProcessingPipeline(llm=llm)
    return pipeline.process(image_path, callbacks)
