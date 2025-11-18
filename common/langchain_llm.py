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
    _processor: Any = PrivateAttr()  # Can be AutoProcessor (Llama) or AutoTokenizer (InternVL3)
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
            processor: Loaded processor (Llama) or tokenizer (InternVL3) instance
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

    def _build_internvl_transform(self, input_size: int = 448):
        """Build InternVL3 image transformation pipeline."""
        import torchvision.transforms as T

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return transform

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Standard InternVL3 find_closest_aspect_ratio."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(self, image: Image.Image, min_num: int = 1, max_num: int = 12,
                           image_size: int = 448, use_thumbnail: bool = False):
        """Standard InternVL3 dynamic_preprocess."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []

        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _get_model_device(self):
        """Get the primary device where the vision model is located."""
        import torch

        # For InternVL3, get device from vision model (critical for multi-GPU)
        try:
            # Try vision model's embeddings first (where image processing happens)
            if hasattr(self._model, 'vision_model') and hasattr(self._model.vision_model, 'embeddings'):
                return next(self._model.vision_model.embeddings.parameters()).device
            # Try vision model directly
            elif hasattr(self._model, 'vision_model'):
                return next(self._model.vision_model.parameters()).device
            # Fallback to first model parameter
            else:
                return next(self._model.parameters()).device
        except Exception:
            # Last resort fallback
            return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _load_internvl_image(self, image: Image.Image, input_size: int = 448, max_num: int = 12):
        """Complete InternVL3 image loading and preprocessing pipeline."""
        import torch

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process into tiles
        images = self._dynamic_preprocess(
            image, min_num=1, max_num=max_num, image_size=input_size, use_thumbnail=True
        )

        # Apply transforms
        transform = self._build_internvl_transform(input_size=input_size)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)

        # InternVL3: Vision model is ALWAYS on cuda:0
        vision_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Detect model dtype (bfloat16 for non-quantized, float16 for quantized)
        try:
            model_dtype = next(self._model.parameters()).dtype
        except Exception:
            # Fallback to bfloat16 for non-quantized models
            model_dtype = torch.bfloat16

        # Convert to correct device AND dtype in one call (critical for multi-GPU)
        pixel_values = pixel_values.to(device=vision_device, dtype=model_dtype)

        return pixel_values

    def _generate_internvl(self, prompt: str, image: Image.Image, **kwargs: Any) -> str:
        """
        Generate text using InternVL3 with .chat() method.

        Args:
            prompt: Text prompt
            image: PIL Image
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text
        """
        # Prepare image for InternVL3 (already on cuda:0 with float16 dtype)
        pixel_values = self._load_internvl_image(image, max_num=12)

        # Create generation config as dictionary (InternVL3 requirement)
        generation_config = {
            'max_new_tokens': kwargs.get("max_new_tokens", self.max_new_tokens),
            'do_sample': kwargs.get("do_sample", self.do_sample),
            'pad_token_id': self._processor.eos_token_id,  # Suppress warnings
        }

        # Add temperature and top_p only if do_sample is True
        if generation_config['do_sample']:
            generation_config['temperature'] = kwargs.get("temperature", self.temperature)
            if self.top_p is not None:
                generation_config['top_p'] = kwargs.get("top_p", self.top_p)

        # InternVL3 requires <image> token in the question
        question = f"<image>\n{prompt}"

        # Use InternVL3's chat method
        response = self._model.chat(
            tokenizer=self._processor,  # InternVL3 uses tokenizer
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            history=None,
            return_history=False
        )

        return response.strip()

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
        # CRITICAL: Pass YAML config parameters to model loader
        # InternVL3 loader needs use_quantization, torch_dtype, device_map, etc.
        # Llama loader uses simpler signature (just model_path)
        model_config_dict = model_config.to_dict()

        # Try to call with full config first (InternVL3 style)
        try:
            import inspect
            loader_signature = inspect.signature(model_loader_func)
            loader_params = set(loader_signature.parameters.keys())

            # Build kwargs for model loader using only recognized parameters
            loader_kwargs = {}
            if 'use_quantization' in loader_params:
                loader_kwargs['use_quantization'] = model_config_dict.get('use_quantization', False)
            if 'torch_dtype' in loader_params:
                loader_kwargs['torch_dtype'] = model_config_dict.get('torch_dtype', 'bfloat16')
            if 'device_map' in loader_params:
                loader_kwargs['device_map'] = model_config_dict.get('device_map', 'auto')
            if 'max_new_tokens' in loader_params:
                loader_kwargs['max_new_tokens'] = model_config_dict.get('max_new_tokens', 2048)
            if 'verbose' in loader_params:
                loader_kwargs['verbose'] = override_kwargs.get("verbose", True)

            # Call model loader with extracted parameters
            model, processor = model_loader_func(model_config.model_id, **loader_kwargs)

        except Exception as e:
            # Fallback to simple signature (Llama style)
            print(f"Warning: Could not call loader with full config: {e}")
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
