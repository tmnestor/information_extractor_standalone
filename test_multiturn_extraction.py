#!/usr/bin/env python3
"""
Test Multi-Turn Bank Statement Extraction (V2)

Tests the new multi-turn extractor that:
1. Detects table structure and column headers
2. Extracts columns one-by-one using actual header names
3. Validates alignment and consistency

Usage:
    # Single image
    python test_multiturn_extraction.py --image evaluation_data/images/image_008.png --model llama-3.2-11b-vision

    # Process all images in a directory with ground truth validation
    python test_multiturn_extraction.py \
        --image-dir evaluation_data/synthetic_bank_images \
        --ground-truth evaluation_data/ground_truth_synthetic_bank.csv \
        --model llama-3.2-11b-vision

    # Process directory without ground truth
    python test_multiturn_extraction.py --image-dir evaluation_data/images --model llama-3.2-11b-vision
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
    """Test multi-turn extraction on bank statement image(s)"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test multi-turn bank statement extraction"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single bank statement image",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing bank statement images",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground truth CSV file for validation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.2-11b-vision",
        help="Model to use (default: llama-3.2-11b-vision)",
    )

    args = parser.parse_args()

    # Determine image paths
    image_paths = []
    if args.image:
        if not Path(args.image).exists():
            console.print(f"[red]Error: Image not found: {args.image}[/red]")
            return 1
        image_paths = [args.image]
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            console.print(f"[red]Error: Directory not found: {args.image_dir}[/red]")
            return 1
        # Get all PNG files in directory
        image_paths = sorted(image_dir.glob("*.png"))
        if not image_paths:
            console.print(f"[red]Error: No PNG images found in {args.image_dir}[/red]")
            return 1
        console.print(f"[cyan]Found {len(image_paths)} images in {args.image_dir}[/cyan]")
    else:
        # Default to single test image
        default_image = "evaluation_data/images/image_008.png"
        if not Path(default_image).exists():
            console.print(f"[red]Error: Default image not found: {default_image}[/red]")
            console.print("[yellow]Use --image or --image-dir to specify images[/yellow]")
            return 1
        image_paths = [default_image]

    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        import csv
        gt_path = Path(args.ground_truth)
        if not gt_path.exists():
            console.print(f"[red]Error: Ground truth file not found: {args.ground_truth}[/red]")
            return 1

        ground_truth = {}
        with gt_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                ground_truth[row['image_file']] = row
        console.print(f"[cyan]Loaded ground truth for {len(ground_truth)} images[/cyan]")

    # Load model and config
    llm = load_model(args.model)
    config = get_yaml_config()

    # Create multi-turn extractor
    extractor = MultiTurnExtractorV2(llm=llm, config=config)

    # Process each image
    for image_path in image_paths:
        image_name = Path(image_path).name

        # Extract bank statement (returns markdown table)
        console.print(f"\n[bold cyan]Processing: {image_path}[/bold cyan]")
        markdown_table = extractor.extract_bank_statement(str(image_path))

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

        # Validate against ground truth if available
        if ground_truth and image_name in ground_truth:
            console.print("\n" + "=" * 80)
            console.print("[bold yellow]Ground Truth Comparison:[/bold yellow]")
            console.print("=" * 80)
            gt = ground_truth[image_name]
            console.print(f"Expected dates: {gt.get('TRANSACTION_DATES', 'N/A')}")
            console.print(f"Expected amounts: {gt.get('TRANSACTION_AMOUNTS_PAID', 'N/A')}")
            console.print(f"Expected descriptions: {gt.get('LINE_ITEM_DESCRIPTIONS', 'N/A')[:100]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
