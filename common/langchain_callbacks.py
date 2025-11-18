"""
LangChain Callbacks for Batch Document Processing

Provides structured callbacks for monitoring document processing pipelines,
replacing scattered rprint() statements with organized, trackable metrics.

Key Features:
- Progress tracking across batch operations
- Token usage monitoring
- Error aggregation and reporting
- Performance metrics collection
- Rich console output integration
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from rich import print as rprint
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn


class BatchProcessingCallback(BaseCallbackHandler):
    """
    Callback handler for tracking batch document processing operations.

    This callback integrates with LangChain chains to provide structured
    monitoring, metrics collection, and progress visualization using Rich.

    Usage:
        >>> from rich.console import Console
        >>>
        >>> console = Console()
        >>> callback = BatchProcessingCallback(console, verbose=True)
        >>>
        >>> # Use with chains
        >>> result = chain.run(input_data, callbacks=[callback])
        >>>
        >>> # Get metrics
        >>> metrics = callback.get_metrics()
        >>> print(f"Total tokens: {metrics['total_tokens']}")
        >>> print(f"Total errors: {metrics['total_errors']}")

    Attributes:
        console: Rich Console instance for output
        verbose: Enable detailed logging
        metrics: Dictionary storing processing metrics
        errors: List of errors encountered
        current_stage: Name of current processing stage
        stage_start_time: Timestamp when current stage started
    """

    def __init__(
        self,
        console: Console,
        verbose: bool = True,
        enable_progress_bar: bool = False,
    ):
        """
        Initialize the callback handler.

        Args:
            console: Rich Console instance for formatted output
            verbose: Enable detailed logging of operations
            enable_progress_bar: Show progress bar for batch operations
        """
        super().__init__()
        self.console = console
        self.verbose = verbose
        self.enable_progress_bar = enable_progress_bar

        # Metrics tracking
        self.metrics: Dict[str, Any] = {
            "total_tokens": 0,
            "total_api_calls": 0,
            "total_errors": 0,
            "total_successes": 0,
            "stages_completed": [],
            "processing_times": {},
        }

        # Error tracking
        self.errors: List[Dict[str, Any]] = []

        # Stage tracking
        self.current_stage: Optional[str] = None
        self.stage_start_time: Optional[datetime] = None

        # Progress bar (if enabled)
        self.progress: Optional[Progress] = None
        self.progress_task: Optional[TaskID] = None

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """
        Called when LLM starts processing.

        Args:
            serialized: Serialized LLM information
            prompts: List of prompts being processed
            **kwargs: Additional arguments
        """
        if self.verbose:
            stage = kwargs.get("invocation_params", {}).get("stage", "unknown")
            self.console.print(f"[cyan]🔄 LLM call started: {stage}[/cyan]")

        self.metrics["total_api_calls"] += 1

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        """
        Called when LLM finishes processing.

        Args:
            response: LLM response containing generations and metadata
            **kwargs: Additional arguments
        """
        # Track token usage if available
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            total_tokens = token_usage.get("total_tokens", 0)

            if total_tokens > 0:
                self.metrics["total_tokens"] += total_tokens

                if self.verbose:
                    self.console.print(
                        f"[green]✅ LLM call completed: {total_tokens} tokens[/green]"
                    )

        self.metrics["total_successes"] += 1

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any
    ) -> None:
        """
        Called when LLM encounters an error.

        Args:
            error: The error that occurred
            **kwargs: Additional arguments
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stage": self.current_stage,
        }

        self.errors.append(error_info)
        self.metrics["total_errors"] += 1

        if self.verbose:
            self.console.print(
                f"[red]❌ LLM error ({error_info['error_type']}): {error_info['error_message']}[/red]"
            )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Called when a chain starts.

        Args:
            serialized: Serialized chain information
            inputs: Chain inputs
            **kwargs: Additional arguments
        """
        # Extract stage name
        stage_name = serialized.get("name", "unknown_chain")
        self.current_stage = stage_name
        self.stage_start_time = datetime.now()

        if self.verbose:
            self.console.print(f"[bold blue]▶️  Starting stage: {stage_name}[/bold blue]")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """
        Called when a chain completes.

        Args:
            outputs: Chain outputs
            **kwargs: Additional arguments
        """
        if self.current_stage and self.stage_start_time:
            # Calculate stage duration
            duration = (datetime.now() - self.stage_start_time).total_seconds()
            self.metrics["processing_times"][self.current_stage] = duration

            # Record completion
            self.metrics["stages_completed"].append({
                "stage": self.current_stage,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            })

            if self.verbose:
                self.console.print(
                    f"[green]✅ Completed stage: {self.current_stage} "
                    f"({duration:.2f}s)[/green]"
                )

        # Reset stage tracking
        self.current_stage = None
        self.stage_start_time = None

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any
    ) -> None:
        """
        Called when a chain encounters an error.

        Args:
            error: The error that occurred
            **kwargs: Additional arguments
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stage": self.current_stage,
            "chain_level": True,
        }

        self.errors.append(error_info)
        self.metrics["total_errors"] += 1

        if self.verbose:
            self.console.print(
                f"[red]❌ Chain error in {self.current_stage}: {error}[/red]"
            )

    def start_batch(self, total_items: int, description: str = "Processing") -> None:
        """
        Start tracking a batch operation.

        Args:
            total_items: Total number of items to process
            description: Description of the batch operation
        """
        if self.enable_progress_bar:
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )
            self.progress.start()
            self.progress_task = self.progress.add_task(
                description, total=total_items
            )
        else:
            if self.verbose:
                self.console.print(
                    f"[bold cyan]📊 Starting batch: {description} "
                    f"({total_items} items)[/bold cyan]"
                )

    def update_batch(self, advance: int = 1) -> None:
        """
        Update batch progress.

        Args:
            advance: Number of items to advance progress by
        """
        if self.enable_progress_bar and self.progress and self.progress_task:
            self.progress.update(self.progress_task, advance=advance)

    def end_batch(self) -> None:
        """Complete batch operation and stop progress tracking."""
        if self.enable_progress_bar and self.progress:
            self.progress.stop()
            self.progress = None
            self.progress_task = None

    def log_custom_metric(self, name: str, value: Any) -> None:
        """
        Log a custom metric.

        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self.metrics:
            self.metrics[name] = []

        if isinstance(self.metrics[name], list):
            self.metrics[name].append(value)
        else:
            self.metrics[name] = value

        if self.verbose:
            self.console.print(f"[dim]📊 Metric: {name} = {value}[/dim]")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.

        Returns:
            Dict[str, Any]: Dictionary containing all metrics
        """
        return {
            **self.metrics,
            "total_stages": len(self.metrics["stages_completed"]),
            "error_rate": (
                self.metrics["total_errors"] /
                max(self.metrics["total_api_calls"], 1)
            ),
        }

    def get_errors(self) -> List[Dict[str, Any]]:
        """
        Get all recorded errors.

        Returns:
            List[Dict[str, Any]]: List of error dictionaries
        """
        return self.errors

    def print_summary(self) -> None:
        """Print a formatted summary of metrics and errors."""
        self.console.rule("[bold green]Processing Summary[/bold green]")

        # Metrics
        metrics = self.get_metrics()
        rprint(f"[cyan]Total API Calls: {metrics['total_api_calls']}[/cyan]")
        rprint(f"[cyan]Total Tokens Used: {metrics['total_tokens']}[/cyan]")
        rprint(f"[green]Successes: {metrics['total_successes']}[/green]")

        if metrics['total_errors'] > 0:
            rprint(f"[red]Errors: {metrics['total_errors']}[/red]")
            rprint(f"[yellow]Error Rate: {metrics['error_rate']:.2%}[/yellow]")

        # Processing times
        if metrics['processing_times']:
            self.console.print("\n[bold blue]Stage Durations:[/bold blue]")
            for stage, duration in metrics['processing_times'].items():
                rprint(f"  {stage}: {duration:.2f}s")

        # Errors
        if self.errors:
            self.console.print("\n[bold red]Errors:[/bold red]")
            for error in self.errors:
                rprint(
                    f"  [{error['timestamp']}] {error['error_type']}: "
                    f"{error['error_message']}"
                )

    def reset(self) -> None:
        """Reset all metrics and error tracking."""
        self.metrics = {
            "total_tokens": 0,
            "total_api_calls": 0,
            "total_errors": 0,
            "total_successes": 0,
            "stages_completed": [],
            "processing_times": {},
        }
        self.errors = []
        self.current_stage = None
        self.stage_start_time = None


class DocumentProcessingCallback(BatchProcessingCallback):
    """
    Specialized callback for document processing with field extraction tracking.

    Extends BatchProcessingCallback with document-specific metrics like
    field extraction counts and accuracy tracking.
    """

    def __init__(
        self,
        console: Console,
        verbose: bool = True,
        enable_progress_bar: bool = False,
    ):
        """
        Initialize document processing callback.

        Args:
            console: Rich Console instance
            verbose: Enable verbose logging
            enable_progress_bar: Show progress bar
        """
        super().__init__(console, verbose, enable_progress_bar)

        # Document-specific metrics
        self.metrics["documents_processed"] = 0
        self.metrics["fields_extracted"] = 0
        self.metrics["fields_matched"] = 0
        self.metrics["document_types"] = {}

    def log_document_processed(
        self,
        document_type: str,
        fields_extracted: int,
        fields_matched: Optional[int] = None,
    ) -> None:
        """
        Log metrics for a processed document.

        Args:
            document_type: Type of document (invoice, receipt, etc.)
            fields_extracted: Number of fields extracted
            fields_matched: Number of fields matching ground truth (if available)
        """
        self.metrics["documents_processed"] += 1
        self.metrics["fields_extracted"] += fields_extracted

        if fields_matched is not None:
            self.metrics["fields_matched"] += fields_matched

        # Track by document type
        if document_type not in self.metrics["document_types"]:
            self.metrics["document_types"][document_type] = {
                "count": 0,
                "total_fields": 0,
            }

        self.metrics["document_types"][document_type]["count"] += 1
        self.metrics["document_types"][document_type]["total_fields"] += fields_extracted

        if self.verbose:
            self.console.print(
                f"[green]✅ Processed {document_type}: "
                f"{fields_extracted} fields extracted[/green]"
            )

    def print_summary(self) -> None:
        """Print document-specific summary."""
        super().print_summary()

        # Document-specific metrics
        self.console.print("\n[bold blue]Document Processing Metrics:[/bold blue]")
        rprint(f"[cyan]Documents Processed: {self.metrics['documents_processed']}[/cyan]")
        rprint(f"[cyan]Total Fields Extracted: {self.metrics['fields_extracted']}[/cyan]")

        if self.metrics['fields_matched'] > 0:
            accuracy = (
                self.metrics['fields_matched'] /
                max(self.metrics['fields_extracted'], 1)
            )
            rprint(f"[green]Field Accuracy: {accuracy:.2%}[/green]")

        # By document type
        if self.metrics['document_types']:
            self.console.print("\n[bold blue]By Document Type:[/bold blue]")
            for doc_type, stats in self.metrics['document_types'].items():
                avg_fields = stats['total_fields'] / max(stats['count'], 1)
                rprint(
                    f"  {doc_type}: {stats['count']} documents, "
                    f"{avg_fields:.1f} avg fields"
                )
