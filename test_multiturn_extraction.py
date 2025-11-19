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

    # Extract bank statement
    console.print(f"\n[bold cyan]Processing: {image_path}[/bold cyan]")
    result = extractor.extract_bank_statement(image_path)

    # Display results
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]Extraction Results:[/bold cyan]")
    console.print("=" * 80)

    console.print(f"\n[green]Structure Type:[/green] {result.structure.structure_type}")
    console.print(
        f"[green]Column Headers:[/green] {' | '.join(result.structure.column_headers)}"
    )
    console.print(f"[green]Total Rows:[/green] {result.row_count}")

    console.print("\n[green]Sample Transactions (first 3):[/green]")
    for i in range(min(3, result.row_count)):
        console.print(f"\n[cyan]Transaction {i+1}:[/cyan]")
        console.print(f"  Date: {result.dates[i]}")
        console.print(f"  Description: {result.descriptions[i]}")
        console.print(f"  Debit: {result.debits[i]}")
        console.print(f"  Credit: {result.credits[i]}")
        console.print(f"  Balance: {result.balances[i]}")

    # Show validation status
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]Validation:[/bold cyan]")
    console.print("=" * 80)

    if result.validation_passed:
        console.print("[green]✅ All validations passed[/green]")
    else:
        console.print(
            f"[yellow]⚠️  {len(result.validation_errors)} validation warnings:[/yellow]"
        )
        for error in result.validation_errors:
            console.print(f"  • {error}")

    # Show compatible output format
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]Compatible Output (Evaluation Format):[/bold cyan]")
    console.print("=" * 80)

    output = result.to_dict()
    console.print("\n[green]TRANSACTION_DATES:[/green]")
    console.print(f"  {output['TRANSACTION_DATES'][:200]}...")

    console.print("\n[green]LINE_ITEM_DESCRIPTIONS:[/green]")
    console.print(f"  {output['LINE_ITEM_DESCRIPTIONS'][:200]}...")

    console.print("\n[green]TRANSACTION_AMOUNTS_PAID:[/green]")
    console.print(f"  {output['TRANSACTION_AMOUNTS_PAID'][:200]}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
