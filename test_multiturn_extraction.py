#!/usr/bin/env python3
"""
Test Multi-Turn Bank Statement Extraction (V2)

Tests the new multi-turn extractor that:
1. Detects table structure and column headers
2. Extracts columns one-by-one using actual header names
3. Validates alignment and consistency

Usage:
    python test_multiturn_extraction.py --image evaluation_data/images/image_008.png --model llama-3.2-11b-vision
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console  # noqa: E402
from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: E402

from common.config import get_yaml_config  # noqa: E402
from common.langchain_llm import VisionLanguageModel  # noqa: E402
from common.llama_model_loader import load_llama_model  # noqa: E402
from common.multiturn_extractor_v2 import MultiTurnExtractorV2  # noqa: E402

console = Console()


def load_model(model_name: str = "llama-3.2-11b-vision"):
    """Load vision-language model."""
    console.print(f"\n[bold cyan]Loading model: {model_name}[/bold cyan]")

    config = get_yaml_config()
    model_config = config.get_model_config(model_name)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Loading {model_name}...", total=None)

        if "llama" in model_name.lower():
            model, processor = load_llama_model(
                model_path=model_config.model_id,
                use_quantization=False,
                device_map=model_config.device_map,
                torch_dtype=model_config.torch_dtype,
                max_new_tokens=model_config.max_new_tokens,
            )
        elif "internvl" in model_name.lower():
            from common.internvl3_model_loader import load_internvl3_model

            model, tokenizer = load_internvl3_model(
                model_path=model_config.model_id,
                use_quantization=False,
                device_map=model_config.device_map,
                torch_dtype=model_config.torch_dtype,
                max_new_tokens=model_config.max_new_tokens,
            )
            processor = tokenizer
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        progress.update(task, completed=True)

    console.print("[green]✅ Model loaded successfully[/green]")

    # Wrap in VisionLanguageModel for LangChain compatibility
    llm = VisionLanguageModel(
        model=model,
        processor=processor,
        model_id=model_name,  # Pass model_id for auto-detection
        max_new_tokens=2000,
    )

    return llm


def main():
    """Test multi-turn extraction on bank statement image"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test multi-turn bank statement extraction"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="evaluation_data/images/image_008.png",
        help="Path to bank statement image (default: image_008.png)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.2-11b-vision",
        help="Model to use (default: llama-3.2-11b-vision)",
    )

    args = parser.parse_args()
    image_path = args.image

    if not Path(image_path).exists():
        console.print(f"[red]Error: Image not found: {image_path}[/red]")
        return 1

    # Load model and config
    llm = load_model(args.model)
    config = get_yaml_config()

    # Create multi-turn extractor
    extractor = MultiTurnExtractorV2(llm=llm, config=config)

    # Extract bank statement (returns markdown table)
    console.print(f"\n[bold cyan]Processing: {image_path}[/bold cyan]")
    markdown_table = extractor.extract_bank_statement(image_path)

    # Display the extracted markdown table
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]Extracted 3-Column Markdown Table:[/bold cyan]")
    console.print("=" * 80)
    console.print()

    # Use rich.markdown to render the table
    from rich.markdown import Markdown
    console.print(Markdown(markdown_table))

    # Also show raw markdown for inspection
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]Raw Markdown (for copying/validation):[/bold cyan]")
    console.print("=" * 80)
    console.print(markdown_table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
