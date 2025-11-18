#!/usr/bin/env python3
"""
Full Integration Test: Phases 1-4 with Real Models and Images

This script demonstrates complete end-to-end extraction using:
- Phase 1: Model-specific prompts
- Phase 2: Bank statement classification
- Phase 3: Enhanced structure-specific prompts
- Phase 4: Multi-turn extraction

Usage:
    # Test with Llama
    python test_full_integration.py --model llama-3.2-11b-vision-8bit --image evaluation_data/images/image_001.png

    # Test with InternVL3
    python test_full_integration.py --model internvl3-2b --image evaluation_data/images/image_001.png

    # Process all images in directory
    python test_full_integration.py --model llama-3.2-11b-vision-8bit --batch evaluation_data/images/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from common.config import get_yaml_config
from common.langchain_llm import VisionLanguageModel
from common.prompt_registry import get_registry

console = Console()


def load_model(model_name: str):
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
            from common.llama_model_loader import load_llama_model

            model, processor = load_llama_model(
                model_id=model_config.model_id,
                quantization_bits=model_config.quantization_bits,
                device_map=model_config.device_map,
            )
        elif "internvl" in model_name.lower():
            from common.internvl3_model_loader import load_internvl3_model

            model, tokenizer = load_internvl3_model(
                model_id=model_config.model_id,
                quantization_bits=model_config.quantization_bits,
                device_map=model_config.device_map,
            )
            processor = tokenizer
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        progress.update(task, completed=True)

    console.print("[green]✅ Model loaded successfully[/green]")
    return model, processor


def classify_document_structure(image_path: Path, llm, model_name: str):
    """Classify bank statement structure (Phase 2)."""
    console.print("\n[bold cyan]Phase 2: Classifying document structure[/bold cyan]")

    registry = get_registry()
    classifier = registry.get_bank_classifier(llm=llm, model_name=model_name)

    result = classifier.classify(image_path=image_path)

    # Display classification results
    table = Table(title="Bank Statement Classification")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Structure Type", result.structure_type)
    table.add_row("Column Count", str(result.column_count) if result.column_count else "N/A")
    table.add_row("Confidence", result.confidence)
    table.add_row("Extraction Approach", result.get_extraction_approach())
    table.add_row("Multi-Turn Recommended", "Yes" if result.requires_multi_turn() else "No")

    console.print(table)

    return result


def extract_with_single_pass(image_path: Path, llm, model_name: str, structure_type: str):
    """Extract using single-pass method with structure-specific prompt (Phase 3)."""
    console.print("\n[bold cyan]Phase 3: Single-pass extraction with structure-specific prompt[/bold cyan]")

    registry = get_registry()

    # Get structure-optimized prompt
    prompt = registry.get_prompt(
        document_type="bank_statement",
        model_name=model_name,
        structure_type=structure_type,
    )

    console.print(f"[yellow]Using prompt optimized for: {structure_type}[/yellow]")

    # Format prompt with image
    from PIL import Image

    image = Image.open(image_path)
    messages = prompt.format_messages(image=image)

    # Invoke model
    with console.status("[bold green]Extracting data..."):
        response = llm.invoke(messages)

    console.print("\n[bold green]Extraction Result:[/bold green]")
    console.print(Panel(response.content, title="Extracted Data", border_style="green"))

    return response.content


def extract_with_multiturn(image_path: Path, llm):
    """Extract using multi-turn method (Phase 4)."""
    console.print("\n[bold cyan]Phase 4: Multi-turn extraction[/bold cyan]")

    registry = get_registry()
    extractor = registry.get_multiturn_extractor(llm=llm)

    console.print("[yellow]Executing 6-step multi-turn workflow...[/yellow]")

    with console.status("[bold green]Extracting columns..."):
        result = extractor.extract_bank_statement(image_path=image_path)

    # Display results
    console.print("\n[bold green]Multi-Turn Extraction Results:[/bold green]")
    console.print(f"Rows extracted: {result.row_count}")
    console.print(f"Validation passed: {'✅' if result.validation_passed else '❌'}")

    if not result.validation_passed:
        console.print("\n[bold red]Validation Errors:[/bold red]")
        for error in result.validation_errors:
            console.print(f"  ❌ {error}")

    # Show sample data
    if result.row_count > 0:
        table = Table(title=f"Sample Transactions (first {min(5, result.row_count)} rows)")
        table.add_column("Date", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Debit", style="red")
        table.add_column("Credit", style="green")
        table.add_column("Balance", style="yellow")

        for i in range(min(5, result.row_count)):
            table.add_row(
                result.dates[i] if i < len(result.dates) else "N/A",
                result.descriptions[i][:50] if i < len(result.descriptions) else "N/A",
                result.debits[i] if i < len(result.debits) else "N/A",
                result.credits[i] if i < len(result.credits) else "N/A",
                result.balances[i] if i < len(result.balances) else "N/A",
            )

        console.print(table)

    return result


def process_document(image_path: Path, model_name: str, model, processor):
    """Process a document using all 4 phases."""
    console.print(Panel.fit(
        f"[bold]Processing Document[/bold]\n"
        f"Image: {image_path.name}\n"
        f"Model: {model_name}",
        border_style="cyan"
    ))

    # Create LangChain LLM wrapper
    llm = VisionLanguageModel(
        model=model,
        processor=processor,
        model_id=model_name,
        max_new_tokens=2000,
        temperature=0.0,
    )

    # Phase 1: Get model-specific prompt info
    console.print("\n[bold cyan]Phase 1: Model-specific prompts[/bold cyan]")
    registry = get_registry()
    model_info = registry.get_model_info(model_name)

    if model_info:
        console.print(f"Prompt style: [green]{model_info.get('style', 'default')}[/green]")
        console.print(f"Verbosity: [green]{model_info.get('verbosity', 'default')}[/green]")

    # For bank statements: classify structure first
    if "bank" in image_path.stem.lower() or "statement" in image_path.stem.lower():
        # Phase 2: Classify structure
        structure_result = classify_document_structure(image_path, llm, model_name)

        # Choose extraction method based on structure
        if structure_result.requires_multi_turn():
            # Phase 4: Multi-turn extraction
            result = extract_with_multiturn(image_path, llm)
        else:
            # Phase 3: Single-pass with structure-specific prompt
            result = extract_with_single_pass(
                image_path, llm, model_name, structure_result.structure_type
            )
    else:
        # For invoices/receipts: use standard single-pass
        console.print("\n[bold cyan]Standard extraction (invoice/receipt)[/bold cyan]")

        prompt = registry.get_prompt(
            document_type="invoice",  # or detect type first
            model_name=model_name,
        )

        from PIL import Image

        image = Image.open(image_path)
        messages = prompt.format_messages(image=image)

        with console.status("[bold green]Extracting data..."):
            response = llm.invoke(messages)

        console.print("\n[bold green]Extraction Result:[/bold green]")
        console.print(Panel(response.content, title="Extracted Data", border_style="green"))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Full integration test with real models and images"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., llama-3.2-11b-vision-8bit, internvl3-2b)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image to process",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to directory with images to process",
    )

    args = parser.parse_args()

    if not args.image and not args.batch:
        console.print("[red]Error: Must specify either --image or --batch[/red]")
        return 1

    try:
        # Load model
        model, processor = load_model(args.model)

        # Process images
        if args.image:
            # Single image
            image_path = Path(args.image)
            if not image_path.exists():
                console.print(f"[red]Error: Image not found: {image_path}[/red]")
                return 1

            process_document(image_path, args.model, model, processor)

        elif args.batch:
            # Batch processing
            batch_dir = Path(args.batch)
            if not batch_dir.exists():
                console.print(f"[red]Error: Directory not found: {batch_dir}[/red]")
                return 1

            image_files = list(batch_dir.glob("*.png")) + list(batch_dir.glob("*.jpg"))

            console.print(f"\n[bold]Processing {len(image_files)} images...[/bold]\n")

            for image_path in image_files:
                process_document(image_path, args.model, model, processor)
                console.print("\n" + "=" * 80 + "\n")

        console.print("\n[bold green]✅ Processing complete![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        import traceback

        console.print(traceback.format_exc())
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
