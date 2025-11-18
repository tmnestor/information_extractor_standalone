"""
LangChain Integration Test with Real Document Images

Tests the complete LangChain pipeline with actual document images and ground truth data.
Compares performance against existing BatchDocumentProcessor.

Usage:
    python tests/test_langchain_integration.py [--verbose] [--max-images N]
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party imports
import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

# Local imports - using try/except for optional dependencies
try:
    from common.langchain_callbacks import DocumentProcessingCallback  # noqa: E402
    from common.langchain_chains import DocumentProcessingPipeline  # noqa: E402
    from common.langchain_llm import LlamaVisionLLM  # noqa: E402
    from common.llama_model_loader_robust import load_llama_model_robust  # noqa: E402
except ImportError as e:
    # These will be imported when needed if not available at top level
    pass

console = Console()


def load_test_data():
    """Load test images and ground truth."""
    console.print("\n[cyan]📂 Loading test data...[/cyan]")

    # Find test images
    eval_dir = project_root / "evaluation_data"
    image_files = sorted(eval_dir.glob("*.png"))

    if not image_files:
        console.print(f"[red]❌ No test images found in {eval_dir}[/red]")
        return None, None

    console.print(f"[green]✅ Found {len(image_files)} test images[/green]")

    # Load ground truth
    ground_truth_path = eval_dir / "ground_truth.csv"
    if not ground_truth_path.exists():
        console.print(f"[yellow]⚠️  No ground truth found at {ground_truth_path}[/yellow]")
        return image_files, None

    ground_truth = pd.read_csv(ground_truth_path)
    console.print(f"[green]✅ Loaded ground truth: {len(ground_truth)} rows[/green]")

    return image_files, ground_truth


def load_model():
    """Load Llama model using existing loader."""
    console.print("\n[cyan]🤖 Loading Llama Vision model...[/cyan]")

    try:
        # Check for model path in config
        model_path = None
        try:
            from common.config import LLAMA_MODEL_PATH

            # Check if config path exists
            if Path(LLAMA_MODEL_PATH).exists():
                model_path = LLAMA_MODEL_PATH
        except (ImportError, AttributeError):
            pass

        # If config path doesn't exist or wasn't found, try fallback paths
        if not model_path:
            possible_paths = [
                "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct",
                "/nfs_share/models/Llama-3.2-11B-Vision-Instruct",
                "/data/model-weights/Llama-3.2-11B-Vision-Instruct",
                str(Path.home() / "models" / "Llama-3.2-11B-Vision-Instruct"),
            ]

            for path in possible_paths:
                if Path(path).exists():
                    model_path = path
                    break

        if not model_path:
            console.print(
                "[red]❌ Could not find Llama model in any known location.[/red]"
            )
            console.print(
                "[yellow]💡 Update common/config.py or provide --model-path[/yellow]"
            )
            return None, None

        console.print(f"[dim]Model path: {model_path}[/dim]")

        # Load model
        start_time = time.time()
        model, processor = load_llama_model_robust(
            model_path,
            use_quantization=True,  # Use 8-bit for memory efficiency
            device_map="auto",
        )
        load_time = time.time() - start_time

        console.print(
            f"[green]✅ Model loaded in {load_time:.1f}s[/green]"
        )

        return model, processor

    except Exception as e:
        console.print(f"[red]❌ Failed to load model: {e}[/red]")
        traceback.print_exc()
        return None, None


def create_langchain_pipeline(model, processor, verbose=False):
    """Create LangChain processing pipeline."""
    console.print("\n[cyan]🔗 Creating LangChain pipeline...[/cyan]")

    try:
        # Wrap model in LangChain LLM
        llm = LlamaVisionLLM(
            model=model,
            processor=processor,
            max_new_tokens=2048,
            temperature=0.0,  # Deterministic for testing
            verbose=verbose,
        )

        # Create callback for monitoring
        callback = DocumentProcessingCallback(
            console=console,
            verbose=verbose,
            enable_progress_bar=True,
        )

        # Create pipeline
        pipeline = DocumentProcessingPipeline(
            llm=llm,
            enable_fixing=False,  # Disable for speed
            verbose=verbose,
        )

        console.print("[green]✅ LangChain pipeline created[/green]")

        return pipeline, callback, llm

    except Exception as e:
        console.print(f"[red]❌ Failed to create pipeline: {e}[/red]")
        traceback.print_exc()
        return None, None, None


def run_pipeline_test(
    pipeline,
    callback,
    llm,
    image_files,
    ground_truth=None,
    max_images=None,
):
    """Run pipeline on test images and collect metrics."""
    console.print("\n[cyan]🚀 Running pipeline test...[/cyan]")

    if max_images:
        image_files = image_files[:max_images]
        console.print(f"[yellow]⚠️  Limited to first {max_images} images[/yellow]")

    # Process batch
    start_time = time.time()

    try:
        results = pipeline.process_batch(
            image_paths=[str(f) for f in image_files],
            callbacks=[callback],
        )
        processing_time = time.time() - start_time

        console.print(
            f"\n[green]✅ Processed {len(results)} images in {processing_time:.1f}s[/green]"
        )
        console.print(
            f"[cyan]⏱️  Average: {processing_time / len(results):.1f}s per image[/cyan]"
        )

        # Collect metrics
        metrics = {
            "total_images": len(results),
            "successful": sum(1 for r in results if r["document_type"] != "error"),
            "failed": sum(1 for r in results if r["document_type"] == "error"),
            "processing_time": processing_time,
            "avg_time_per_image": processing_time / len(results),
            "callback_metrics": callback.get_metrics(),
            "llm_metrics": llm.get_metrics(),
        }

        return results, metrics

    except Exception as e:
        console.print(f"[red]❌ Pipeline test failed: {e}[/red]")
        traceback.print_exc()
        return None, None


# Import existing extraction cleaner and Decimal type from common
from decimal import Decimal  # noqa: E402

from common.extraction_cleaner import ExtractionCleaner  # noqa: E402


def serialize_pydantic_to_llm_format(field_name: str, value) -> str:
    """
    Convert Pydantic model types to LLM output format strings.

    This prepares types for ExtractionCleaner which expects LLM string output.

    Args:
        field_name: Name of the field
        value: Pydantic field value (List, Decimal, bool, str, etc.)

    Returns:
        String representation that ExtractionCleaner can process
    """
    # Handle NOT_FOUND
    if value == "NOT_FOUND" or value is None:
        return "NOT_FOUND"

    # Handle lists - convert to comma-separated (cleaner will convert to pipes)
    if isinstance(value, list):
        if not value:
            return "NOT_FOUND"
        # Convert each item to string
        str_items = []
        for item in value:
            if isinstance(item, Decimal):
                str_items.append(f"${str(item)}")
            else:
                str_items.append(str(item))
        return ", ".join(str_items)

    # Handle Decimal - convert to string with $ for monetary fields
    if isinstance(value, Decimal):
        return f"${str(value)}"

    # Handle boolean - convert to lowercase string
    if isinstance(value, bool):
        return str(value).lower()

    # Everything else - just convert to string
    return str(value)


def evaluate_accuracy(results, ground_truth, verbose=False):
    """Evaluate extraction accuracy against ground truth."""
    console.print("\n[cyan]📊 Evaluating accuracy...[/cyan]")

    if ground_truth is None:
        console.print("[yellow]⚠️  No ground truth available, skipping accuracy evaluation[/yellow]")
        return None

    # Create lookup by image filename
    gt_lookup = {row["image_file"]: row for _, row in ground_truth.iterrows()}

    # Initialize cleaner for converting Pydantic types to CSV format
    cleaner = ExtractionCleaner(debug=False)

    # Calculate field-level accuracy
    field_matches = {}
    field_totals = {}
    mismatches = []  # Track mismatches for debugging

    for result in results:
        if result["document_type"] == "error":
            continue

        image_name = Path(result["image_path"]).name

        if image_name not in gt_lookup:
            continue

        gt_row = gt_lookup[image_name]

        # Convert Pydantic model to dict, serialize types to LLM format, then clean
        extracted_raw = result["extracted_data"].model_dump()

        # Serialize Python types to LLM output format strings
        extracted_serialized = {
            field: serialize_pydantic_to_llm_format(field, value)
            for field, value in extracted_raw.items()
        }

        # Now clean using ExtractionCleaner (expects LLM string output)
        extracted_cleaned_dict = cleaner.clean_extraction_dict(extracted_serialized)

        if verbose:
            console.print(f"\n[bold cyan]🔍 {image_name}[/bold cyan]")

        # Compare each field
        for field_name, extracted_value in extracted_cleaned_dict.items():
            if field_name not in gt_row:
                continue

            # Ground truth is already in CSV format from pandas
            gt_value = str(gt_row[field_name])

            # Initialize counters
            if field_name not in field_matches:
                field_matches[field_name] = 0
                field_totals[field_name] = 0

            field_totals[field_name] += 1

            # Compare cleaned extracted value with ground truth
            # Both should now be in CSV string format
            if extracted_value == gt_value:
                field_matches[field_name] += 1
                if verbose:
                    console.print(f"  [green]✓ {field_name}[/green]")
            else:
                if verbose:
                    console.print(f"  [red]✗ {field_name}[/red]")
                    console.print(f"    Expected: [yellow]{gt_value[:100]}[/yellow]")
                    console.print(f"    Got:      [red]{extracted_value[:100]}[/red]")

                mismatches.append({
                    "image": image_name,
                    "field": field_name,
                    "expected": gt_value[:100],
                    "got": extracted_value[:100],
                })

    # Calculate accuracy per field
    field_accuracy = {
        field: (field_matches.get(field, 0) / field_totals[field] * 100)
        if field in field_totals and field_totals[field] > 0
        else 0.0
        for field in field_totals
    }

    # Overall accuracy
    total_matches = sum(field_matches.values())
    total_fields = sum(field_totals.values())
    overall_accuracy = (
        (total_matches / total_fields * 100) if total_fields > 0 else 0.0
    )

    accuracy_metrics = {
        "overall_accuracy": overall_accuracy,
        "field_accuracy": field_accuracy,
        "total_fields_compared": total_fields,
        "total_matches": total_matches,
    }

    console.print(
        f"[green]✅ Overall Accuracy: {overall_accuracy:.1f}%[/green]"
    )
    console.print(
        f"[cyan]   ({total_matches}/{total_fields} fields matched)[/cyan]"
    )

    return accuracy_metrics


def print_summary(results, metrics, accuracy_metrics=None):
    """Print comprehensive test summary."""
    console.rule("[bold green]LangChain Integration Test Summary[/bold green]")

    # Show errors first if any
    errors = [r for r in results if r.get("document_type") == "error"]
    if errors:
        console.print("\n[bold red]❌ Errors Encountered[/bold red]")
        for i, error in enumerate(errors[:5], 1):  # Show first 5 errors
            console.print(f"[red]{i}. {error['image_name']}: {error.get('error', 'Unknown error')}[/red]")
        if len(errors) > 5:
            console.print(f"[dim]... and {len(errors) - 5} more errors[/dim]")

    # Processing metrics
    console.print("\n[bold cyan]📊 Processing Metrics[/bold cyan]")
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Images Processed", str(metrics["total_images"]))
    table.add_row("Successful", str(metrics["successful"]))
    table.add_row("Failed", str(metrics["failed"]))
    table.add_row("Total Time", f"{metrics['processing_time']:.1f}s")
    table.add_row("Avg per Image", f"{metrics['avg_time_per_image']:.1f}s")

    # LLM metrics
    llm_metrics = metrics["llm_metrics"]
    table.add_row("Total Tokens", str(llm_metrics["total_tokens_used"]))
    table.add_row("Avg Tokens/Image", str(llm_metrics["avg_tokens_per_call"]))

    console.print(table)

    # Accuracy metrics
    if accuracy_metrics:
        console.print("\n[bold cyan]🎯 Accuracy Metrics[/bold cyan]")
        acc_table = Table()
        acc_table.add_column("Metric", style="cyan")
        acc_table.add_column("Value", style="green")

        acc_table.add_row(
            "Overall Accuracy",
            f"{accuracy_metrics['overall_accuracy']:.1f}%",
        )
        acc_table.add_row(
            "Total Fields Compared",
            str(accuracy_metrics["total_fields_compared"]),
        )
        acc_table.add_row(
            "Total Matches",
            str(accuracy_metrics["total_matches"]),
        )

        console.print(acc_table)

        # Top/bottom performing fields
        field_acc = accuracy_metrics["field_accuracy"]
        if field_acc:
            sorted_fields = sorted(
                field_acc.items(), key=lambda x: x[1], reverse=True
            )

            console.print("\n[bold cyan]📈 Top 5 Fields (Accuracy)[/bold cyan]")
            for field, acc in sorted_fields[:5]:
                console.print(f"  {field}: {acc:.1f}%")

            console.print("\n[bold cyan]📉 Bottom 5 Fields (Accuracy)[/bold cyan]")
            for field, acc in sorted_fields[-5:]:
                console.print(f"  {field}: {acc:.1f}%")

    # Callback metrics
    callback_metrics = metrics["callback_metrics"]
    console.print("\n[bold cyan]📞 Callback Metrics[/bold cyan]")
    cb_table = Table()
    cb_table.add_column("Metric", style="cyan")
    cb_table.add_column("Value", style="green")

    cb_table.add_row("API Calls", str(callback_metrics["total_api_calls"]))
    cb_table.add_row("Successes", str(callback_metrics["total_successes"]))
    cb_table.add_row("Errors", str(callback_metrics["total_errors"]))
    cb_table.add_row("Documents Processed", str(callback_metrics["documents_processed"]))
    cb_table.add_row("Fields Extracted", str(callback_metrics["fields_extracted"]))

    console.print(cb_table)

    console.rule("[bold green]Test Complete[/bold green]")


def main():
    """Run integration test."""
    parser = argparse.ArgumentParser(
        description="LangChain Integration Test with Real Documents"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Limit number of images to process",
    )
    args = parser.parse_args()

    console.rule("[bold green]LangChain Integration Test[/bold green]")
    console.print("[dim]Testing complete pipeline with real document images[/dim]")

    # Step 1: Load test data
    image_files, ground_truth = load_test_data()
    if not image_files:
        console.print("[red]❌ No test data available[/red]")
        return 1

    # Step 2: Load model
    model, processor = load_model()
    if model is None:
        console.print("[red]❌ Failed to load model[/red]")
        return 1

    # Step 3: Create pipeline
    pipeline, callback, llm = create_langchain_pipeline(
        model, processor, verbose=args.verbose
    )
    if pipeline is None:
        console.print("[red]❌ Failed to create pipeline[/red]")
        return 1

    # Step 4: Run pipeline test
    results, metrics = run_pipeline_test(
        pipeline, callback, llm, image_files, ground_truth, args.max_images
    )
    if results is None:
        console.print("[red]❌ Pipeline test failed[/red]")
        return 1

    # Step 5: Evaluate accuracy
    accuracy_metrics = evaluate_accuracy(results, ground_truth, verbose=args.verbose)

    # Step 6: Print summary
    print_summary(results, metrics, accuracy_metrics)

    console.print("\n[bold green]🎉 Integration test complete![/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
