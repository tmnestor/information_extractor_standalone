#!/usr/bin/env python3
"""
Test markdown table extraction from bank statements.

This approach asks the vision model to simply OCR the table into markdown format,
then we can programmatically parse and filter it.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import yaml  # noqa: E402
from rich.console import Console  # noqa: E402

from common.image_processor import process_image_for_model  # noqa: E402
from common.model_loader import load_model  # noqa: E402

# Use IPython display if available, fallback to print
try:
    from IPython.display import Markdown, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    from rich.markdown import Markdown as RichMarkdown

console = Console()


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
    console.print(f"[cyan]Loading model: {model_name}[/cyan]")
    model, processor = load_model(model_name)

    # Load table extraction prompt
    with Path("config/prompts.yaml").open() as f:
        config = yaml.safe_load(f)

    table_prompt = config["table_extraction_template"]

    # Process image
    image = process_image_for_model(image_path, processor)

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
    """Test markdown extraction on image_003.png"""
    image_path = "evaluation_data/images/image_003.png"

    if not Path(image_path).exists():
        console.print(f"[red]Error: Image not found: {image_path}[/red]")
        return 1

    # Extract markdown table
    markdown_table = extract_table_as_markdown(image_path)

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
