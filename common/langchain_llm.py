"""
LangChain BaseChatModel Wrapper for Vision-Language Models

Provides a LangChain v1.0 compatible interface for vision-language models,
enabling integration with LangChain chains, prompts, and utilities while
maintaining support for vision inputs.

Key Features:
- LangChain BaseChatModel compatibility (v1.0)
- Vision-language model support (text + images)
- Multi-modal message handling
- YAML-based configuration
- Token usage tracking
- Supports Llama-3.2-Vision and InternVL3 models
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from PIL import Image
from pydantic import Field, PrivateAttr
from transformers import AutoProcessor


class VisionLanguageModel(BaseChatModel):
    """
    LangChain v1.0 wrapper for vision-language models.

    This wrapper enables the use of vision-language models (Llama-3.2-Vision,
    InternVL3) with LangChain's chain and prompt management system while
    maintaining vision capabilities.

    Usage:
        >>> from common.config import get_yaml_config, ModelConfig
        >>> from common.llama_model_loader import load_llama_model
        >>>
        >>> # Load model configuration from YAML
        >>> config = get_yaml_config()
        >>> model_config = config.get_model_config("llama-3.2-11b-vision-8bit")
        >>>
        >>> # Load model using existing loader
        >>> model, processor = load_llama_model(model_config.model_id)
        >>>
        >>> # Wrap in LangChain interface
        >>> llm = VisionLanguageModel(
        ...     model=model,
        ...     processor=processor,
        ...     model_config=model_config
        ... )
        >>>
        >>> # Use with multi-modal messages
        >>> from langchain_core.messages import HumanMessage
        >>> message = HumanMessage(
        ...     content=[
        ...         {"type": "text", "text": "Describe this image"},
        ...         {"type": "image_url", "image_url": {"url": "/path/to/image.png"}}
        ...     ]
        ... )
        >>> result = llm.invoke([message])

    Note:
        Images can be provided as file paths or base64-encoded data URLs.
        The model handles both Llama chat templates and InternVL3 formats.
    """

    # Public fields (validated by Pydantic)
    model_id: str = Field(description="Model identifier")
    max_new_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    do_sample: bool = Field(default=False, description="Whether to use sampling")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter")
    verbose: bool = Field(default=False, description="Enable verbose logging")

    # Private attributes (not validated by Pydantic)
    _model: Any = PrivateAttr()
    _processor: AutoProcessor = PrivateAttr()
    _total_tokens_used: int = PrivateAttr(default=0)
    _api_calls: int = PrivateAttr(default=0)
    _is_llama: bool = PrivateAttr(default=False)

    def __init__(
        self,
        model: Any,
        processor: AutoProcessor,
        model_id: str = "",
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the LangChain wrapper for vision-language models.

        Args:
            model: Loaded model instance (Llama or InternVL3)
            processor: Loaded processor instance
            model_id: Model identifier for tracking
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Nucleus sampling parameter
            verbose: Enable verbose logging
            **kwargs: Additional arguments passed to parent BaseChatModel
        """
        # Initialize parent with public fields
        super().__init__(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
            **kwargs
        )

        # Set private attributes
        self._model = model
        self._processor = processor
        self._total_tokens_used = 0
        self._api_calls = 0

        # Detect if this is a Llama model (needs chat template)
        self._is_llama = "llama" in model_id.lower() or "Llama" in str(type(model))

        if self.verbose:
            print(f"✅ Initialized VisionLanguageModel: {model_id}")
            print(f"   Model type: {'Llama (chat template)' if self._is_llama else 'InternVL3 (direct)'}")

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "vision_language_model"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for this LLM."""
        return {
            "model_id": self.model_id,
            "model_type": "Llama-3.2-Vision" if self._is_llama else "InternVL3",
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _format_messages(self, messages: List[BaseMessage]) -> tuple[str, Optional[Image.Image]]:
        """
        Format LangChain messages into prompt text and image.

        Args:
            messages: List of LangChain messages (SystemMessage, HumanMessage, AIMessage)

        Returns:
            tuple: (prompt_text, image)

        Raises:
            ValueError: If no image is found in messages
        """
        text_parts = []
        image = None

        for message in messages:
            if isinstance(message, SystemMessage):
                text_parts.append(message.content)
            elif isinstance(message, HumanMessage):
                # Extract text and image from message
                if isinstance(message.content, str):
                    text_parts.append(message.content)
                elif isinstance(message.content, list):
                    # Handle multi-modal content
                    for item in message.content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item["text"])
                            elif item.get("type") == "image_url":
                                # Load image from URL or path
                                image_source = item.get("image_url", {}).get("url", "")
                                if image_source:
                                    image = self._load_image(image_source)
                        elif isinstance(item, str):
                            text_parts.append(item)
            elif isinstance(message, AIMessage):
                # Include previous AI responses for context
                text_parts.append(message.content)

        prompt = "\n\n".join(text_parts)
        return prompt, image

    def _load_image(self, image_source: str) -> Image.Image:
        """
        Load image from file path or base64 data URL.

        Args:
            image_source: File path or base64-encoded data URL

        Returns:
            PIL Image object in RGB mode

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image_source format is invalid
        """
        if image_source.startswith("data:image"):
            # Handle base64 encoded images
            header, data = image_source.split(",", 1)
            image_data = base64.b64decode(data)
            return Image.open(BytesIO(image_data)).convert("RGB")
        else:
            # Handle file paths
            image_path = Path(image_source)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            return Image.open(image_path).convert("RGB")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Core generation method for BaseChatModel interface.

        This method is called by LangChain chains for chat-based generation.
        Supports multi-modal messages with both text and images.

        Args:
            messages: List of LangChain messages (SystemMessage, HumanMessage, AIMessage)
            stop: Stop sequences (not currently supported)
            run_manager: Callback manager for monitoring
            **kwargs: Additional generation parameters

        Returns:
            ChatResult: LangChain chat result with generated message

        Raises:
            ValueError: If no image is found in messages
        """
        # Format messages into prompt and extract image
        prompt, image = self._format_messages(messages)

        if image is None:
            raise ValueError(
                "No image found in messages. Vision-language models require an image.\n"
                "Add image using: HumanMessage(content=[{'type': 'image_url', "
                "'image_url': {'url': '/path/to/image.png'}}, {'type': 'text', 'text': 'prompt'}])"
            )

        # Generate based on model type
        if self._is_llama:
            generated_text = self._generate_llama(prompt, image, **kwargs)
        else:
            generated_text = self._generate_internvl(prompt, image, **kwargs)

        # Update metrics
        self._api_calls += 1

        # Create chat generation result
        ai_message = AIMessage(content=generated_text)
        generation = ChatGeneration(message=ai_message)

        return ChatResult(generations=[generation])

    def _generate_llama(self, prompt: str, image: Image.Image, **kwargs: Any) -> str:
        """
        Generate text using Llama-3.2-Vision with chat template.

        Args:
            prompt: Text prompt
            image: PIL Image
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text
        """
        # Prepare multimodal input with chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Apply chat template
        input_text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # Tokenize with image
        inputs = self._processor(
            images=image,
            text=input_text,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)

        # Generate
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
                do_sample=kwargs.get("do_sample", self.do_sample),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
            )

        # Decode only the new tokens
        generated_text = self._processor.decode(
            output[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Update token count
        self._total_tokens_used += output.shape[1]

        return generated_text.strip()

    def _generate_internvl(self, prompt: str, image: Image.Image, **kwargs: Any) -> str:
        """
        Generate text using InternVL3 (no chat template needed).

        Args:
            prompt: Text prompt
            image: PIL Image
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text
        """
        # Prepare inputs (InternVL3 uses direct format)
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self._model.device)

        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": kwargs.get("do_sample", self.do_sample),
        }
        if self.top_p is not None:
            gen_kwargs["top_p"] = kwargs.get("top_p", self.top_p)

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **gen_kwargs
            )

        # Decode
        generated_text = self._processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0]

        # Remove the prompt from output if present
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()

        # Update token count
        self._total_tokens_used += output_ids.shape[1]

        return generated_text.strip()

    def invoke_with_image(
        self,
        prompt: str,
        image_path: str,
        **kwargs: Any,
    ) -> str:
        """
        Convenience method for simple image + text generation.

        This is a backward-compatible helper that wraps the BaseChatModel interface
        for easy use with single image + prompt inputs.

        Args:
            prompt: Text prompt describing the task
            image_path: Path to the image file
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text response

        Example:
            >>> result = llm.invoke_with_image(
            ...     prompt="Extract fields from this invoice",
            ...     image_path="/path/to/invoice.png"
            ... )
        """
        # Create multi-modal message
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_path}}
            ]
        )

        # Use standard invoke method
        result = self.invoke([message], **kwargs)

        # Extract text from AIMessage
        return result.content if hasattr(result, 'content') else str(result)

    def get_metrics(self) -> Dict[str, int]:
        """
        Get usage metrics for this LLM instance.

        Returns:
            Dict[str, int]: Metrics including tokens used and API calls
        """
        return {
            "total_tokens_used": self._total_tokens_used,
            "api_calls": self._api_calls,
            "avg_tokens_per_call": (
                self._total_tokens_used // self._api_calls
                if self._api_calls > 0
                else 0
            ),
        }

    def reset_metrics(self) -> None:
        """Reset usage metrics to zero."""
        self._total_tokens_used = 0
        self._api_calls = 0


class VisionLanguageModelFactory:
    """
    Factory for creating VisionLanguageModel instances with YAML configuration.
    """

    @staticmethod
    def from_yaml_config(
        model_name: Optional[str] = None,
        model_loader_func: Optional[Any] = None,
        **override_kwargs
    ) -> VisionLanguageModel:
        """
        Create VisionLanguageModel using YAML configuration.

        Args:
            model_name: Name of model config in models.yaml (uses default if None)
            model_loader_func: Optional function to load model (model_path) -> (model, processor)
            **override_kwargs: Override specific config parameters

        Returns:
            VisionLanguageModel: Configured LangChain wrapper

        Example:
            >>> from common.llama_model_loader import load_llama_model
            >>>
            >>> # Load with YAML config
            >>> llm = VisionLanguageModelFactory.from_yaml_config(
            ...     model_name="llama-3.2-11b-vision-8bit",
            ...     model_loader_func=load_llama_model
            ... )
        """
        # Import here to avoid circular dependency
        from common.config import get_yaml_config

        # Get YAML configuration
        yaml_config = get_yaml_config()
        model_config = yaml_config.get_model_config(model_name)

        # Load model using provided loader or default
        if model_loader_func is None:
            raise ValueError(
                "model_loader_func is required. Provide a function like "
                "load_llama_model or load_internvl3_model"
            )

        # Load model and processor
        model, processor = model_loader_func(model_config.model_id)

        # Merge YAML config with overrides
        config_dict = model_config.to_dict()
        config_dict.update(override_kwargs)

        return VisionLanguageModel(
            model=model,
            processor=processor,
            model_id=model_config.model_id,
            max_new_tokens=config_dict.get("max_new_tokens", 2048),
            do_sample=config_dict.get("do_sample", False),
            temperature=config_dict.get("temperature", 0.0),
            top_p=config_dict.get("top_p"),
            verbose=override_kwargs.get("verbose", False),
        )

    @staticmethod
    def from_loaded_model(
        model: Any,
        processor: AutoProcessor,
        model_id: str = "",
        **kwargs
    ) -> VisionLanguageModel:
        """
        Create VisionLanguageModel from already-loaded model and processor.

        Args:
            model: Loaded model instance (Llama or InternVL3)
            processor: Loaded processor instance
            model_id: Model identifier for tracking
            **kwargs: Additional configuration parameters

        Returns:
            VisionLanguageModel: Configured LangChain wrapper

        Example:
            >>> from common.llama_model_loader import load_llama_model
            >>>
            >>> model, processor = load_llama_model("/path/to/model")
            >>> llm = VisionLanguageModelFactory.from_loaded_model(
            ...     model=model,
            ...     processor=processor,
            ...     model_id="llama-3.2-11b-vision",
            ...     max_new_tokens=2000
            ... )
        """
        return VisionLanguageModel(
            model=model,
            processor=processor,
            model_id=model_id,
            **kwargs
        )
