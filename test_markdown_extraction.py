#!/usr/bin/env python3
"""
Test markdown table extraction from bank statements.

This approach asks the vision model to simply OCR the table into markdown format,
then we can programmatically parse and filter it.
"""

import sys
from pathlib import Path

import yaml
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console  # noqa: E402
from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: E402

from common.config import get_yaml_config  # noqa: E402
from common.llama_model_loader import load_llama_model  # noqa: E402

# Use IPython display if available, fallback to print
try:
    from IPython.display import Markdown, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    from rich.markdown import Markdown as RichMarkdown

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

        model, processor = load_llama_model(
            model_path=model_config.model_id,
            use_quantization=False,
            device_map=model_config.device_map,
            torch_dtype=model_config.torch_dtype,
            max_new_tokens=model_config.max_new_tokens,
        )

        progress.update(task, completed=True)

    console.print("[green]✅ Model loaded successfully[/green]")
    return model, processor


def extract_table_as_markdown(image_path: str, model_name: str = "llama-3.2-11b-vision"):
    """
    Extract bank statement table as markdown.

    Args:
        image_path: Path to bank statement image
        model_name: Model to use for extraction

    Returns:
        Markdown table string
    """
    # Load model
    model, processor = load_model(model_name)

    # Load table extraction prompt
    with Path("config/prompts.yaml").open() as f:
        config = yaml.safe_load(f)

    table_prompt = config["table_extraction_template"]

    # Load image
    image = Image.open(image_path)

    # Create prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": table_prompt},
            ],
        }
    ]

    # Get response
    console.print(f"\n[cyan]Extracting table from: {image_path}[/cyan]")
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
    response = processor.decode(output[0], skip_special_tokens=True)

    # Extract just the markdown table from response
    # Response format: <prompt>assistant\n<markdown table>
    if "assistant" in response:
        markdown_table = response.split("assistant")[-1].strip()
    else:
        markdown_table = response.strip()

    return markdown_table


def main():
    """Test markdown extraction on bank statement image"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract bank statement table as markdown and filter withdrawals"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="evaluation_data/images/image_003.png",
        help="Path to bank statement image (default: image_003.png)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.2-11b-vision",
        help="Model to use (default: llama-3.2-11b-vision)",
    )
    parser.add_argument(
        "--parse",
        action="store_true",
        help="Parse and filter the extracted markdown to show withdrawal transactions only",
    )

    args = parser.parse_args()
    image_path = args.image

    if not Path(image_path).exists():
        console.print(f"[red]Error: Image not found: {image_path}[/red]")
        return 1

    # Extract markdown table
    console.print(f"\n[bold cyan]Processing: {image_path}[/bold cyan]")
    markdown_table = extract_table_as_markdown(image_path, args.model)

    # Display the extracted markdown
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]Extracted Markdown Table:[/bold cyan]")
    console.print("=" * 80)

    if HAS_IPYTHON:
        # Use IPython display (works in Jupyter/IPython)
        display(Markdown(markdown_table))
    else:
        # Fallback to rich markdown
        console.print(RichMarkdown(markdown_table))

    # Also print raw markdown for inspection
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]Raw Markdown (for copying):[/bold cyan]")
    console.print("=" * 80)
    console.print(markdown_table)

    # Parse and filter if requested
    if args.parse:
        from common.markdown_table_parser import process_bank_statement_markdown

        console.print("\n" + "=" * 80)
        console.print("[bold cyan]Parsed & Filtered (Withdrawals Only):[/bold cyan]")
        console.print("=" * 80)

        result = process_bank_statement_markdown(markdown_table)

        console.print("\n[green]TRANSACTION_DATES:[/green]")
        console.print(f"  {result['TRANSACTION_DATES']}")

        console.print("\n[green]LINE_ITEM_DESCRIPTIONS:[/green]")
        console.print(f"  {result['LINE_ITEM_DESCRIPTIONS']}")

        console.print("\n[green]TRANSACTION_AMOUNTS_PAID:[/green]")
        console.print(f"  {result['TRANSACTION_AMOUNTS_PAID']}")

        console.print("\n[cyan]Stats:[/cyan]")
        console.print(f"  Total rows in table: {result['_total_rows']}")
        console.print(f"  Withdrawal transactions: {result['_withdrawal_rows']}")
        console.print(f"  Credit transactions filtered: {result['_filtered_out']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
