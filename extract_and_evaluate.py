#!/usr/bin/env python3
"""
Extract and Evaluate: Vision-Language Model Document Extraction with Field-by-Field Evaluation

This production-ready script provides complete end-to-end document extraction and evaluation:

Features:
- Phase 1: Model-specific prompts (Llama vs InternVL3)
- Phase 2: Bank statement classification (10+ structural formats)
- Phase 3: Enhanced structure-specific prompts with layout guidance
- Phase 4: Multi-turn extraction for complex tables
- Field-by-field evaluation with complete value display (no truncation)
- F1 metrics (precision, recall, TP/FP/FN) for each field
- Batch processing with aggregate statistics
- JSON export for further analysis

Usage:
    # Single image extraction
    python extract_and_evaluate.py --model llama-3.2-11b-vision --image evaluation_data/images/image_001.png

    # Single image with evaluation
    python extract_and_evaluate.py --model llama-3.2-11b-vision --image evaluation_data/images/image_001.png --evaluate

    # Batch extraction with evaluation
    python extract_and_evaluate.py --model llama-3.2-11b-vision --batch evaluation_data/images/ --evaluate

    # Debug mode: show only mismatches
    python extract_and_evaluate.py --model llama-3.2-11b-vision --batch evaluation_data/images/ --evaluate --show-mismatches-only

    # Save evaluation results to JSON
    python extract_and_evaluate.py --model llama-3.2-11b-vision --batch evaluation_data/images/ --evaluate --save-results results.json

Models Supported:
    - llama-3.2-11b-vision: Llama 3.2 Vision 11B (step-by-step prompts)
    - llama-3.2-11b-vision-8bit: Llama 3.2 Vision 11B with 8-bit quantization
    - internvl3-2b: InternVL3 2B (concise prompts)
    - internvl3-4b: InternVL3 4B

Document Types:
    - Invoices: 14 fields (ABN, supplier, line items, GST, total)
    - Receipts: 14 fields (same as invoices)
    - Bank Statements: 5 fields (dates, transactions, amounts)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from common.config import get_yaml_config
from common.evaluation_metrics import (
    calculate_field_accuracy_f1,
    load_ground_truth,
)
from common.langchain_llm import VisionLanguageModel
from common.prompt_registry import get_registry

console = Console()


def detect_document_type(image_path: Path, llm) -> str:
    """
    Detect document type using specialized detection prompt.

    Args:
        image_path: Path to image file
        llm: VisionLanguageModel instance

    Returns:
        Detected document type: "invoice", "receipt", or "bank_statement"
    """
    import yaml

    # Load detection prompts
    detection_yaml = Path(__file__).parent / "config" / "document_type_detection.yaml"
    with detection_yaml.open('r') as f:
        detection_config = yaml.safe_load(f)

    # Get complex detection prompt (works slightly better)
    prompt_text = detection_config["prompts"]["detection_complex"]["prompt"]

    # Run detection
    response = llm.invoke_with_image(
        prompt=prompt_text,
        image_path=str(image_path)
    )

    # Normalize response using type mappings
    response_lower = response.strip().lower()
    type_mappings = detection_config["type_mappings"]

    for variant, canonical in type_mappings.items():
        if variant in response_lower:
            return canonical.lower()  # Return lowercase: invoice, receipt, bank_statement

    # Fallback
    return detection_config["settings"]["fallback_type"].lower()


def display_extraction_prompt(prompt_text: str, prompt_type: str = "Extraction") -> None:
    """
    Display extraction prompt in a formatted panel.

    Args:
        prompt_text: The actual prompt text being sent to the model
        prompt_type: Type of prompt (e.g., "Extraction", "Detection", "Classification")
    """
    # Truncate very long prompts for display
    max_display_length = 2000
    display_text = prompt_text if len(prompt_text) <= max_display_length else (
        prompt_text[:max_display_length] + f"\n\n... (truncated, total length: {len(prompt_text)} chars)"
    )

    console.print(Panel(
        display_text,
        title=f"[bold yellow]{prompt_type} Prompt[/bold yellow]",
        border_style="yellow",
        expand=False,
    ))


def parse_extraction_output(text: str) -> Dict[str, str]:
    """
    Parse LLM output text into structured field dictionary.

    Handles format:
        FIELD_NAME: value
        FIELD_NAME: value1 | value2 | value3

    Args:
        text: Raw LLM output text

    Returns:
        Dictionary mapping field names to values
    """
    fields = {}
    current_field = None
    current_value = []

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check if line starts a new field (FIELD_NAME: value)
        if ":" in line and line.split(":")[0].isupper():
            # Save previous field if exists
            if current_field:
                fields[current_field] = " ".join(current_value).strip()

            # Start new field
            parts = line.split(":", 1)
            current_field = parts[0].strip()
            current_value = [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
        elif current_field:
            # Continuation of previous field (multi-line value)
            current_value.append(line)

    # Save last field
    if current_field:
        fields[current_field] = " ".join(current_value).strip()

    # Convert empty values to "NOT_FOUND"
    for field, value in fields.items():
        if not value or value.lower() in ["", "n/a", "none", "null"]:
            fields[field] = "NOT_FOUND"

    return fields


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
                model_path=model_config.model_id,
                use_quantization=False,  # No quantization
                device_map=model_config.device_map,
                torch_dtype=model_config.torch_dtype,
                max_new_tokens=model_config.max_new_tokens,
            )
        elif "internvl" in model_name.lower():
            from common.internvl3_model_loader import load_internvl3_model

            model, tokenizer = load_internvl3_model(
                model_path=model_config.model_id,
                use_quantization=False,  # No quantization
                device_map=model_config.device_map,
                torch_dtype=model_config.torch_dtype,
                max_new_tokens=model_config.max_new_tokens,
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

    # Get prompt text (handle both string and list content)
    messages = prompt.format_messages(image="<image>")

    text_parts = []
    for msg in messages:
        if isinstance(msg.content, str):
            text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            # Extract text from multi-modal content
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)

    prompt_text = "\n\n".join(text_parts)

    # Display the actual prompt being used
    display_extraction_prompt(prompt_text, prompt_type="Bank Statement Extraction")

    # Invoke model with image
    with console.status("[bold green]Extracting data..."):
        response = llm.invoke_with_image(
            prompt=prompt_text,
            image_path=str(image_path)
        )

    console.print("\n[bold green]Extraction Result:[/bold green]")
    console.print(Panel(response, title="Extracted Data", border_style="green"))

    return response


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


def display_field_comparison(
    field_name: str,
    extracted: str,
    ground_truth: str,
    metrics: Dict,
) -> None:
    """
    Display single field comparison with COMPLETE values (no truncation).

    Args:
        field_name: Name of the field being compared
        extracted: Extracted value (full, no truncation)
        ground_truth: Ground truth value (full, no truncation)
        metrics: Dict with f1_score, precision, recall, etc.
    """
    f1_score = metrics.get("f1_score", 0.0)
    match_status = "✅" if f1_score == 1.0 else "⚠️" if f1_score >= 0.5 else "❌"

    # Color code based on match quality
    if f1_score == 1.0:
        status_color = "green"
        status_text = "EXACT MATCH"
    elif f1_score >= 0.8:
        status_color = "yellow"
        status_text = "FUZZY MATCH"
    elif f1_score > 0.0:
        status_color = "orange"
        status_text = "PARTIAL MATCH"
    else:
        status_color = "red"
        status_text = "MISMATCH"

    # Display field header
    console.print(
        f"\n{match_status} [bold]{field_name}[/bold] "
        f"([{status_color}]F1: {f1_score:.2f} - {status_text}[/{status_color}])"
    )

    # Create side-by-side panels for extracted vs ground truth
    extracted_panel = Panel(
        extracted if extracted else "NOT_FOUND",
        title="Extracted",
        border_style="green" if f1_score == 1.0 else "red",
        expand=False,
        width=60,
    )

    ground_truth_panel = Panel(
        ground_truth if ground_truth else "NOT_FOUND",
        title="Ground Truth",
        border_style="cyan",
        expand=False,
        width=60,
    )

    console.print(Columns([extracted_panel, ground_truth_panel]))

    # Show detailed metrics if not perfect match
    if f1_score < 1.0 and f1_score > 0.0:
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        console.print(
            f"  [dim]Precision: {precision:.2f} | Recall: {recall:.2f} | "
            f"TP: {metrics.get('tp', 0)} | FP: {metrics.get('fp', 0)} | FN: {metrics.get('fn', 0)}[/dim]"
        )


def display_detailed_evaluation(
    image_name: str,
    extracted: Dict[str, str],
    ground_truth: Dict[str, str],
    document_type: str,
    show_mismatches_only: bool = False,
) -> Dict:
    """
    Display complete field-by-field evaluation with FULL values.

    Args:
        image_name: Name of the image file
        extracted: Extracted fields dictionary
        ground_truth: Ground truth fields dictionary
        document_type: Type of document (invoice, receipt, bank_statement)
        show_mismatches_only: Only show fields that don't match perfectly

    Returns:
        Dict with evaluation metrics
    """
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]Field-by-Field Evaluation: {image_name}[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]")

    # Get document-specific fields to evaluate
    from common.config import get_document_type_fields, get_validation_only_fields

    # Get all fields for this document type
    fields_to_evaluate = get_document_type_fields(document_type)

    # Skip validation-only fields for bank statements
    if document_type.lower() == "bank_statement":
        validation_only = get_validation_only_fields()
        fields_to_evaluate = [f for f in fields_to_evaluate if f not in validation_only]

    # Evaluate each field
    total_f1 = 0.0
    perfect_matches = 0
    field_metrics = {}

    for field_name in fields_to_evaluate:
        extracted_value = extracted.get(field_name, "NOT_FOUND")
        gt_value = ground_truth.get(field_name, "NOT_FOUND")

        # Calculate field accuracy with F1 metrics
        metrics = calculate_field_accuracy_f1(
            {field_name: extracted_value},
            {field_name: gt_value},
            field_name
        )

        f1_score = metrics.get("f1_score", 0.0)
        total_f1 += f1_score

        if f1_score == 1.0:
            perfect_matches += 1

        field_metrics[field_name] = metrics

        # Display based on filter
        if show_mismatches_only and f1_score == 1.0:
            continue  # Skip perfect matches

        display_field_comparison(field_name, extracted_value, gt_value, metrics)

    # Summary statistics
    avg_accuracy = total_f1 / len(fields_to_evaluate) if fields_to_evaluate else 0.0

    console.print(f"\n[bold]{'─'*80}[/bold]")
    console.print(
        f"[bold]Summary:[/bold] {perfect_matches}/{len(fields_to_evaluate)} fields perfect "
        f"([{'green' if avg_accuracy >= 0.9 else 'yellow' if avg_accuracy >= 0.7 else 'red'}]"
        f"{avg_accuracy*100:.1f}% accuracy[/])"
    )

    return {
        "image_name": image_name,
        "accuracy": avg_accuracy,
        "perfect_matches": perfect_matches,
        "total_fields": len(fields_to_evaluate),
        "field_metrics": field_metrics,
    }


def display_batch_summary(all_results: List[Dict]) -> None:
    """
    Display aggregate evaluation summary across all images.

    Args:
        all_results: List of evaluation result dicts from display_detailed_evaluation
    """
    if not all_results:
        return

    console.print(f"\n[bold green]{'='*80}[/bold green]")
    console.print("[bold green]BATCH EVALUATION SUMMARY[/bold green]")
    console.print(f"[bold green]{'='*80}[/bold green]\n")

    # Per-image summary table
    table = Table(title=f"Results for {len(all_results)} Images")
    table.add_column("Image", style="cyan")
    table.add_column("Accuracy", style="white")
    table.add_column("Perfect Matches", style="white")
    table.add_column("Worst Field", style="white")

    for result in all_results:
        # Find worst performing field
        worst_field = None
        worst_f1 = 1.0
        for field_name, metrics in result["field_metrics"].items():
            f1 = metrics.get("f1_score", 0.0)
            if f1 < worst_f1:
                worst_f1 = f1
                worst_field = field_name

        accuracy_color = "green" if result["accuracy"] >= 0.9 else "yellow" if result["accuracy"] >= 0.7 else "red"

        table.add_row(
            result["image_name"],
            f"[{accuracy_color}]{result['accuracy']*100:.1f}%[/{accuracy_color}]",
            f"{result['perfect_matches']}/{result['total_fields']}",
            f"{worst_field} ({worst_f1:.2f})" if worst_field else "-",
        )

    console.print(table)

    # Overall statistics
    overall_accuracy = sum(r["accuracy"] for r in all_results) / len(all_results)
    total_perfect = sum(r["perfect_matches"] for r in all_results)
    total_fields = sum(r["total_fields"] for r in all_results)

    console.print("\n[bold]Overall Statistics:[/bold]")
    console.print(f"  Total Images: {len(all_results)}")
    console.print(
        f"  Average Accuracy: [{'green' if overall_accuracy >= 0.9 else 'yellow' if overall_accuracy >= 0.7 else 'red'}]"
        f"{overall_accuracy*100:.1f}%[/]"
    )
    console.print(f"  Perfect Field Matches: {total_perfect}/{total_fields} ({total_perfect/total_fields*100:.1f}%)")

    # Best and worst images
    best_image = max(all_results, key=lambda x: x["accuracy"])
    worst_image = min(all_results, key=lambda x: x["accuracy"])

    console.print(f"\n[bold]Best Performer:[/bold] {best_image['image_name']} ({best_image['accuracy']*100:.1f}%)")
    console.print(f"[bold]Worst Performer:[/bold] {worst_image['image_name']} ({worst_image['accuracy']*100:.1f}%)")


def process_document(
    image_path: Path, model_name: str, model, processor, show_extraction: bool = True
) -> Dict:
    """
    Process a document using all 4 phases.

    Args:
        image_path: Path to image file
        model_name: Name of model to use
        model: Loaded model instance
        processor: Loaded processor instance
        show_extraction: Whether to display extraction results

    Returns:
        Dict with image_name, extracted fields, raw_response, document_type
    """
    if show_extraction:
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
        do_sample=False,  # Greedy decoding (deterministic)
    )

    # Phase 0: Detect document type using specialized prompt
    if show_extraction:
        console.print("\n[bold cyan]Phase 0: Document Type Detection[/bold cyan]")

    document_type = detect_document_type(image_path, llm)

    if show_extraction:
        doc_type_display = document_type.replace("_", " ").title()
        console.print(f"Detected type: [green]{doc_type_display}[/green]")

    # Phase 1: Get model-specific prompt info
    if show_extraction:
        console.print("\n[bold cyan]Phase 1: Model-specific prompts[/bold cyan]")
    registry = get_registry()
    model_info = registry.get_model_info(model_name)

    if show_extraction and model_info:
        console.print(f"Prompt style: [green]{model_info.get('style', 'default')}[/green]")
        console.print(f"Verbosity: [green]{model_info.get('verbosity', 'default')}[/green]")

    raw_response = None

    # For bank statements: classify structure first
    if document_type == "bank_statement":
        # Phase 2: Classify structure
        structure_result = classify_document_structure(image_path, llm, model_name)

        # Choose extraction method based on structure
        if structure_result.requires_multi_turn():
            # Phase 4: Multi-turn extraction
            result = extract_with_multiturn(image_path, llm)
            # Convert multiturn result to text format
            raw_response = f"Multi-turn extraction: {result.row_count} rows"
        else:
            # Phase 3: Single-pass with structure-specific prompt
            raw_response = extract_with_single_pass(
                image_path, llm, model_name, structure_result.structure_type
            )
    else:
        # For invoices/receipts: use standard single-pass
        if show_extraction:
            console.print(f"\n[bold cyan]Standard extraction ({document_type})[/bold cyan]")

        prompt = registry.get_prompt(
            document_type=document_type,
            model_name=model_name,
        )

        # Get prompt text (ChatPromptTemplate returns list with SystemMessage, HumanMessage)
        # We just need the text content
        messages = prompt.format_messages(image="<image>")

        # Extract text from messages (content can be string or list)
        text_parts = []
        for msg in messages:
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            elif isinstance(msg.content, list):
                # Extract text from multi-modal content
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item["text"])
                    elif isinstance(item, str):
                        text_parts.append(item)

        prompt_text = "\n\n".join(text_parts)

        # Display the actual prompt being used
        if show_extraction:
            doc_type_title = document_type.replace("_", " ").title()
            display_extraction_prompt(prompt_text, prompt_type=f"{doc_type_title} Extraction")

        with console.status("[bold green]Extracting data..."):
            # Use convenience method that handles image formatting
            raw_response = llm.invoke_with_image(
                prompt=prompt_text,
                image_path=str(image_path)
            )

        if show_extraction:
            console.print("\n[bold green]Extraction Result:[/bold green]")
            console.print(Panel(raw_response, title="Extracted Data", border_style="green"))

    # Parse extraction output into structured fields
    extracted_fields = parse_extraction_output(raw_response) if raw_response else {}

    return {
        "image_name": image_path.name,
        "extracted": extracted_fields,
        "raw_response": raw_response,
        "document_type": document_type,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract and Evaluate: Vision-language model document extraction with field-by-field evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image extraction
  python extract_and_evaluate.py --model llama-3.2-11b-vision --image evaluation_data/images/image_001.png

  # Batch with evaluation
  python extract_and_evaluate.py --model llama-3.2-11b-vision --batch evaluation_data/images/ --evaluate

  # Debug mismatches only
  python extract_and_evaluate.py --model llama-3.2-11b-vision --batch evaluation_data/images/ --evaluate --show-mismatches-only

  # Save results to JSON
  python extract_and_evaluate.py --model llama-3.2-11b-vision --batch evaluation_data/images/ --evaluate --save-results results.json
        """
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
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Enable evaluation mode (compare extracted vs ground truth)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="evaluation_data/ground_truth.csv",
        help="Path to ground truth CSV file (default: evaluation_data/ground_truth.csv)",
    )
    parser.add_argument(
        "--show-mismatches-only",
        action="store_true",
        help="Only show fields that don't match perfectly (for debugging)",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="Save evaluation results to JSON file",
    )

    args = parser.parse_args()

    if not args.image and not args.batch:
        console.print("[red]Error: Must specify either --image or --batch[/red]")
        return 1

    try:
        # Load ground truth if evaluation mode enabled
        ground_truth = None
        if args.evaluate:
            gt_path = Path(args.ground_truth)
            if not gt_path.exists():
                console.print(f"[red]Error: Ground truth file not found: {gt_path}[/red]")
                return 1
            ground_truth = load_ground_truth(str(gt_path))
            console.print(f"[green]✅ Loaded ground truth from {gt_path}[/green]")

        # Load model
        model, processor = load_model(args.model)

        # Collect evaluation results
        all_eval_results = []

        # Process images
        if args.image:
            # Single image
            image_path = Path(args.image)
            if not image_path.exists():
                console.print(f"[red]Error: Image not found: {image_path}[/red]")
                return 1

            result = process_document(
                image_path, args.model, model, processor, show_extraction=not args.evaluate
            )

            # Evaluate if enabled
            if args.evaluate:
                # Use stem (without extension) for robust ground truth lookup
                image_stem = Path(result["image_name"]).stem
                if image_stem not in ground_truth:
                    console.print(f"[yellow]Warning: No ground truth for {result['image_name']} (stem: {image_stem})[/yellow]")
                else:
                    eval_result = display_detailed_evaluation(
                        result["image_name"],
                        result["extracted"],
                        ground_truth[image_stem],
                        result["document_type"],
                        args.show_mismatches_only,
                    )
                    all_eval_results.append(eval_result)

        elif args.batch:
            # Batch processing
            batch_dir = Path(args.batch)
            if not batch_dir.exists():
                console.print(f"[red]Error: Directory not found: {batch_dir}[/red]")
                return 1

            image_files = sorted(list(batch_dir.glob("*.png")) + list(batch_dir.glob("*.jpg")))

            console.print(f"\n[bold]Processing {len(image_files)} images...[/bold]\n")

            for image_path in image_files:
                result = process_document(
                    image_path, args.model, model, processor, show_extraction=not args.evaluate
                )

                # Evaluate if enabled
                if args.evaluate:
                    # Use stem (without extension) for robust ground truth lookup
                    image_stem = Path(result["image_name"]).stem
                    if image_stem not in ground_truth:
                        console.print(f"[yellow]Warning: No ground truth for {result['image_name']} (stem: {image_stem})[/yellow]")
                    else:
                        eval_result = display_detailed_evaluation(
                            result["image_name"],
                            result["extracted"],
                            ground_truth[image_stem],
                            result["document_type"],
                            args.show_mismatches_only,
                        )
                        all_eval_results.append(eval_result)
                elif not args.evaluate:
                    console.print("\n" + "=" * 80 + "\n")

        # Display batch summary if evaluation enabled
        if args.evaluate and len(all_eval_results) > 1:
            display_batch_summary(all_eval_results)

        # Save results if requested
        if args.save_results and all_eval_results:
            save_path = Path(args.save_results)
            with save_path.open("w") as f:
                json.dump(all_eval_results, f, indent=2, default=str)
            console.print(f"\n[green]✅ Saved evaluation results to {save_path}[/green]")

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
