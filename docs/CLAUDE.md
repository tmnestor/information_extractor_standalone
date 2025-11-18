# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

### Conda Environment
```bash
# Create and activate environment
conda env create -f environment.yml
conda activate vision_notebooks

# Or use automated setup script
source unified_setup.sh
```

### Key Dependencies
- **transformers==4.45.2** (pinned for Llama-3.2-Vision compatibility)
- **torch>=2.0.0** + torchvision (CUDA recommended)
- **accelerate** (device mapping)
- **bitsandbytes** (8-bit quantization)
- **timm>=0.9.0** + **einops>=0.6.0** (InternVL3 requirements)

### Model Setup Requirements
**CRITICAL**: Update model paths in notebooks and scripts before running:
- **Llama path**: `/path/to/Llama-3.2-11B-Vision-Instruct`
- **InternVL3 path**: `/path/to/InternVL3-2B` or `/path/to/InternVL3-8B`
- Check `common/config.py` for production paths

### V100 Optimization Status
**IMPLEMENTED**: Both Llama-3.2-Vision-11B and InternVL3-8B include V100 optimizations:
- **Unified GPU optimization module** in `common/gpu_optimization.py`
- **ResilientGenerator** with multi-tier OOM fallback strategies
- **Memory fragmentation detection** and automatic cleanup
- **Critical fix**: InternVL3 method path selection (chat() vs generate())

## Development Workflow - Three-Machine Setup

### Machine Roles
**CRITICAL**: This project uses a three-machine collaborative workflow:

1. **Mac M1 (Local Development)** - User's primary machine
   - Where user interacts with Claude Code
   - Code editing, ruff checks, git operations
   - Cannot run GPU models locally

2. **H200 GPU Machine (Testing)** - High-spec testing environment
   - Dual H200 GPUs for model testing
   - Primary testing and validation environment
   - Where model performance is verified

3. **V100 Machine (Production)** - User's work deployment target
   - Final deployment destination
   - V100-specific optimizations required
   - Where validated changes are ported after H200 testing

### Development Commands

### Testing/Running
**CRITICAL**: Claude Code cannot run model loading locally due to hardware limitations.

#### Local Code Testing (Mac M1)
When running Python code locally for testing imports, schema validation, or non-GPU tasks:
```bash
# IMPORTANT: Use the full path to the conda environment's Python
/opt/homebrew/Caskroom/miniforge/base/envs/unified_vision_processor/bin/python script.py

# Or for inline testing:
/opt/homebrew/Caskroom/miniforge/base/envs/unified_vision_processor/bin/python -c "
import sys
sys.path.insert(0, '/Users/tod/Desktop/LMM_POC')
from common.schema_loader import FieldSchema
loader = FieldSchema()
print('Schema loaded:', loader.total_fields)
"
```

#### Remote Model Execution (H200/V100)
```bash
# REMOTE EXECUTION REQUIRED for model testing:
python llama_keyvalue.py        # Must run on H200 machine first
python internvl3_keyvalue.py    # Must run on H200 machine first

# Development workflow:
# 1. Claude Code + User (Mac M1): Edit code, run ruff checks, commit changes
# 2. User (H200 Machine): Test changes, validate performance
# 3. User (V100 Machine): Port validated changes to production
# 4. Claude Code: Address any issues based on test results

# Launch Jupyter notebooks (local development only)
jupyter notebook
# Select kernel: "Python (Vision Notebooks)"

# Check GPU memory usage during processing (remote only)
nvidia-smi
```

### Code Quality
```bash
# MANDATORY: All Python code must pass ruff checks before committing
ruff check *.py models/*.py common/*.py

# Apply automatic fixes and formatting
ruff check . --fix
ruff format .
```

**CRITICAL**: This project has a ruff git hook configured. All `*.py` files must pass `ruff check *.py` before any commit. Failure to pass ruff checks will prevent commits from being accepted.

## Architecture Overview

### Core Structure
The codebase follows a **modular processor pattern** comparing two vision-language models:

```
LMM_POC/
├── common/                    # Shared utilities
│   ├── config.py             # Centralized configuration & model paths
│   ├── evaluation_utils.py   # Parsing, image discovery, evaluation
│   └── reporting.py          # Output generation & metrics
├── models/                   # Model-specific processors
│   ├── llama_processor.py    # Llama-3.2-Vision implementation
│   └── internvl3_processor.py # InternVL3-2B implementation  
├── *.py                      # Main execution scripts
└── *.ipynb                   # Interactive notebooks
```

### Key Design Patterns

#### 1. Unified Processor Interface
Both `LlamaProcessor` and `InternVL3Processor` implement:
- `process_single_image(image_path)` → extraction results dict
- `process_image_batch(image_files)` → (results, statistics) tuple
- `get_extraction_prompt()` → model-optimized prompt

#### 2. Shared Configuration
`common/config.py` centralizes:
- **EXTRACTION_FIELDS**: 25 business document fields (ABN, TOTAL, etc.)
- **Model paths**: Production vs development paths (commented alternatives)
- **Output patterns**: Consistent file naming across models
- **Evaluation thresholds**: Deployment readiness metrics

#### 3. Model-Specific Optimizations
- **Llama**: Complex chat template, conversation artifact cleaning
- **InternVL3**: Dynamic preprocessing with tile-based approach

### Data Flow
1. **Image Discovery** → `evaluation_utils.discover_images()`
2. **Model Processing** → `Processor.process_image_batch()`
3. **Response Parsing** → `evaluation_utils.parse_extraction_response()`
4. **Evaluation** → Ground truth comparison & accuracy calculation
5. **Reporting** → CSV outputs, executive summaries, deployment checklists

## LangChain Integration Architecture (NEW)

### Overview
The project now includes a **modular LangChain-based pipeline** as an alternative to the monolithic `BatchDocumentProcessor`. This provides improved maintainability, type safety, and testability.

**Status**: ✅ Complete and ready for testing on remote machines
**Documentation**: See `LANGCHAIN_USAGE_GUIDE.md` for detailed usage examples
**Progress**: See `LANGCHAIN_INTEGRATION_PROGRESS.md` for full implementation details

### Key Dependencies (Added)
- **langchain>=0.1.0** - Core framework for chains and prompts
- **langchain-core>=0.1.0** - Base components
- **langchain-community>=0.0.20** - Additional utilities
- **pydantic>=2.0.0** - Type-safe extraction schemas

### New Modular Components

#### 1. LLM Wrapper (`common/langchain_llm.py`)
```python
from common.langchain_llm import LlamaVisionLLM

# Wrap existing model in LangChain interface
llm = LlamaVisionLLM(
    model=model,
    processor=processor,
    max_new_tokens=2000,
    verbose=True
)

# Use for vision tasks
result = llm.generate_with_image(
    prompt="Extract invoice data",
    image_path="/path/to/invoice.png"
)
```

**Features**:
- LangChain `LLM` base class compatibility
- Vision-language support (text + images)
- Metrics tracking (tokens, API calls)
- Compatible with all LangChain chains

#### 2. Pydantic Schemas (`common/extraction_schemas.py`)
```python
from common.extraction_schemas import InvoiceExtraction, get_schema_for_document_type

# Type-safe extraction
invoice = InvoiceExtraction(
    DOCUMENT_TYPE="INVOICE",
    BUSINESS_ABN="12345678901",
    TOTAL_AMOUNT="$123.45",  # Auto-parsed to Decimal
    LINE_ITEM_DESCRIPTIONS="Item 1 | Item 2",  # Auto-parsed to list
)

# Access with type safety
total: Decimal = invoice.TOTAL_AMOUNT
items: List[str] = invoice.LINE_ITEM_DESCRIPTIONS
```

**Schemas Available**:
- `InvoiceExtraction` (14 fields)
- `ReceiptExtraction` (14 fields)
- `BankStatementExtraction` (5 fields)
- `UniversalExtraction` (19 fields)

**Benefits**:
- Automatic type coercion and validation
- Replaces ~400 lines of custom parsing code
- IDE autocomplete for fields
- Validation errors at parse time

#### 3. Output Parsers (`common/langchain_parsers.py`)
```python
from common.langchain_parsers import DocumentExtractionParser

# Create parser for document type
parser = DocumentExtractionParser(
    document_type="invoice",
    llm=llm,
    enable_fixing=True,  # Self-healing with LLM
    verbose=True
)

# Parse model output to Pydantic model
result = parser.parse_with_fixing(model_output)
print(result.TOTAL_AMOUNT)  # Typed access
```

**Features**:
- Multiple parsing strategies (JSON, plain text, markdown cleaning)
- Self-healing parser (retries with LLM on errors)
- Document-type-aware parsing
- Fallback strategies for robustness

#### 4. Prompt Management (`common/langchain_prompts.py`)
```python
from common.langchain_prompts import LangChainPromptManager

# Dynamic prompt generation
manager = LangChainPromptManager()

# Get extraction prompt for document type
prompt = manager.get_extraction_prompt("invoice")

# Fields are injected dynamically from schema
messages = prompt.format_messages()
```

**Features**:
- Dynamic field injection (no hardcoded field lists)
- Document-type-specific instructions
- Prompt composition and reuse
- Backward compatible with existing YAML prompts

#### 5. Callbacks (`common/langchain_callbacks.py`)
```python
from common.langchain_callbacks import DocumentProcessingCallback
from rich.console import Console

# Create callback for monitoring
callback = DocumentProcessingCallback(
    console=Console(),
    verbose=True,
    enable_progress_bar=True
)

# Use with pipeline
result = pipeline.process(image_path, callbacks=[callback])

# Get metrics
metrics = callback.get_metrics()
print(f"Tokens: {metrics['total_tokens']}")
print(f"Documents: {metrics['documents_processed']}")
```

**Features**:
- Structured metrics collection
- Progress tracking with Rich console
- Error aggregation
- Stage-by-stage timing

#### 6. Processing Chains (`common/langchain_chains.py`)
```python
from common.langchain_chains import DocumentProcessingPipeline, create_pipeline

# Create complete pipeline
pipeline = create_pipeline(llm, enable_fixing=True, verbose=True)

# Process single document
result = pipeline.process("/path/to/invoice.png")

# Process batch
results = pipeline.process_batch(image_paths)
```

**Chains Available**:
- `DocumentDetectionChain` - Identify document type
- `FieldExtractionChain` - Extract structured fields
- `DocumentProcessingPipeline` - End-to-end processing

### LangChain vs Traditional Architecture

#### Before (Monolithic)
```
BatchDocumentProcessor (1354 lines)
  ├─ Direct model.generate() calls
  ├─ Hardcoded chat templates
  ├─ Custom parsing (780 lines)
  ├─ Scattered logging
  └─ Static YAML prompts
```

#### After (Modular)
```
LlamaVisionLLM (305 lines) - Model wrapper
LangChainPromptManager (305 lines) - Dynamic prompts
DocumentExtractionParser (363 lines) - Type-safe parsing
BatchProcessingCallback (387 lines) - Structured monitoring
Pydantic Schemas (421 lines) - Extraction models
DocumentProcessingPipeline (458 lines) - Orchestration
```

**Benefits**:
- ~54% reduction in parsing code complexity
- Type safety with Pydantic validation
- Each component independently testable
- Easier model swapping (Llama ↔ InternVL3)
- Better error handling and self-healing

### Quick Start with LangChain

```python
# 1. Load model (existing code)
from common.llama_model_loader_robust import load_llama_model_robust
model, processor = load_llama_model_robust("/path/to/model")

# 2. Create LangChain components
from common.langchain_llm import LlamaVisionLLM
from common.langchain_chains import create_pipeline

llm = LlamaVisionLLM(model=model, processor=processor)
pipeline = create_pipeline(llm, verbose=True)

# 3. Process documents
result = pipeline.process("/path/to/invoice.png")

# 4. Access typed results
print(f"Type: {result['document_type']}")
print(f"Total: {result['extracted_data'].TOTAL_AMOUNT}")
print(f"Tokens: {result['metrics']['total_tokens_used']}")
```

### Migration Path

**Current State**: Both architectures coexist
- Legacy: `BatchDocumentProcessor` (still functional)
- New: LangChain pipeline (recommended for new code)

**Migration Steps**:
1. Install dependencies: `conda env update -f environment.yml`
2. Test LangChain pipeline with sample documents
3. Compare accuracy and performance
4. Gradually migrate notebooks to new pipeline
5. Deprecate old code once validated

**Compatibility**: Can use both systems simultaneously during transition.

### Testing LangChain Components

```bash
# Phase 1 tests (LLM wrapper, callbacks)
python tests/test_langchain_phase1.py

# Phase 2 tests (schemas, parsers, prompts)
python tests/test_langchain_phase2.py

# Full integration tests (requires GPU)
python tests/test_langchain_integration.py  # Coming soon
```

### When to Use LangChain vs Traditional

**Use LangChain Pipeline When**:
- Building new extraction workflows
- Need type-safe extraction results
- Want self-healing parsers
- Require structured monitoring/metrics
- Planning to swap models (Llama ↔ InternVL3)

**Use Traditional BatchDocumentProcessor When**:
- Working with existing evaluated code
- No need for new features
- Minimizing changes during production runs

### Related Documentation
- **Detailed Usage**: `LANGCHAIN_USAGE_GUIDE.md`
- **Implementation Progress**: `LANGCHAIN_INTEGRATION_PROGRESS.md`
- **Field Definitions**: `config/field_definitions.yaml`

## Task-Specific Information

### Key-Value Extraction Tasks
- **Input**: Business document images (invoices, statements, etc.)
- **Output**: 25 structured fields per document
- **Evaluation**: Against ground truth CSV with exact/fuzzy matching
- **Metrics**: Field-level accuracy, response completeness, content coverage

### VQA (Visual Question Answering) Tasks
- **Simple format**: Image + natural language question → answer
- **Use case**: Interactive document analysis, content understanding
- **Models support different response styles**: Llama (detailed), InternVL3 (concise)

### Hardware Requirements
- **Minimum**: 16GB system RAM, any GPU
- **Llama recommended**: 16GB+ VRAM (or 8-bit quantization)
- **InternVL3 recommended**: 4GB+ VRAM (memory efficient)

### Model Performance Characteristics
| Model | Parameters | Memory | Strengths |
|-------|------------|---------|-----------|
| **Llama-3.2-Vision** | 11B | ~22GB VRAM | Detailed responses, built-in preprocessing |
| **InternVL3-2B** | 2B | ~4GB VRAM | Memory efficient, fast inference, simple API |

## Common Operations

### Running Evaluations
**REMOTE EXECUTION ONLY**: All model evaluation must be run on remote GPU hardware.

```bash
# Full evaluation pipeline (REMOTE ONLY - uses ground truth data)
python llama_keyvalue.py        # Requires GPU server with model access
python internvl3_keyvalue.py    # Requires GPU server with model access

# Outputs generated (remote):
# - {model}_batch_extraction_{timestamp}.csv
# - {model}_evaluation_results_{timestamp}.json  
# - {model}_executive_summary_{timestamp}.md

# Claude Code workflow:
# 1. Prepare/fix code locally with ruff checks
# 2. User runs evaluation remotely  
# 3. User reports results back to Claude Code for analysis
```

### Notebook Development
```bash
# Start Jupyter with correct kernel
jupyter notebook
# Open: llama_VQA.ipynb, internvl3_VQA.ipynb, etc.
# Ensure: Kernel → Python (Vision Notebooks)
```

### Notebook Execution Workflow
**CRITICAL**: After Claude updates any notebook, the user ALWAYS follows this exact workflow:

1. **Sync to Remote Server**: Copy updated notebook to remote GPU server
2. **Restart Kernel**: In Jupyter, select "Kernel → Restart Kernel"
3. **Run All Cells**: Select "Cell → Run All" (or "Run All Cells" from toolbar)

**Important**: Claude should NEVER assume notebooks will be run cell-by-cell or that the kernel state is preserved. All notebook updates must be designed to work when "Run All" is executed on a fresh kernel.

### Path Configuration Updates
When deploying or changing model locations, update:
1. **Development**: Direct path edits in notebooks
2. **Production**: Toggle commented paths in `common/config.py`

### GPU Memory Optimization
Both models include 8-bit quantization support:
- **Enabled by default** in model loading configurations
- **Monitor usage**: `nvidia-smi` during processing
- **CPU fallback**: Models work on CPU (slower performance)

## Vision-Language Model Prompt Guidelines

### CRITICAL RULE: Never Use Actual Image Content in Prompts
**NEVER include actual image content, file names, or specific data from images in prompts.**

❌ **WRONG - Do not do this:**
```
"Extract transactions from this bank statement showing:
- EFTPOS Withdrawal PIZZA HUT: $97.95 
- Return/Refund JB HI-FI: $168.34"
```

✅ **CORRECT - Generic instructions only:**
```
"Extract ALL transactions from this bank statement.
If multiple transactions occur on the same date, extract each as a separate row.
Do not combine transaction descriptions."
```

**Rationale:** 
- Prompts must be generic and reusable across different images
- Including specific image content creates overfitting to particular examples
- Generic prompts ensure consistent extraction behavior across all documents
- Avoids contaminating the model's training with specific test data

## Code and Output Style Guidelines

### CRITICAL RULE: Avoid Verbose Self-Congratulation
**NEVER include excessive boasting, production readiness assessments, or verbose self-congratulation in code outputs.**

❌ **WRONG - Avoid verbose boasting:**
```python
rprint("[bold green]🏭 V100 PRODUCTION READINESS ASSESSMENT[/bold green]")
rprint("✅ V100-Optimized Quantization: BitsAndBytesConfig with CPU offload and vision skip")
rprint("✅ Memory Management: 32MB CUDA blocks, fragmentation detection, comprehensive cleanup")
rprint("✅ OOM Protection: ResilientGenerator with 6-tier fallback system")
# ... 20 more lines of self-congratulation
rprint("[bold green]🎉 V100 Production Optimization Complete![/bold green]")
rprint("The notebook now uses your established V100 best practices with:")
rprint("✓ Proper quantization according to V100_MEMORY_STRATEGIES.md")
# ... more verbose boasting
```

✅ **CORRECT - Concise, factual status:**
```python
rprint("[green]✅ V100-optimized extractor initialized[/green]")
rprint("[cyan]Features: ResilientGenerator, memory cleanup, fragmentation detection[/cyan]")
console.rule("[bold green]Testing Complete[/bold green]")
```

**Rationale:**
- Users find excessive boasting annoying and unprofessional
- Keep status messages concise and factual
- Focus on actual functionality rather than self-praise
- Verbose "production readiness assessments" add no value
- Clean, minimal output is more professional and user-friendly
- ALWAYS FIX ROOT CAUSES NOT JUST THE SYMPTOMS!
- run "ruff check --fix *.py" after every update to a python file
- The project is concerned with comparing the efficacy and efficiency of the Llama and InternVL3 Vision Language Models for Information Extraction from Business Documents.
- Use the conda env /opt/homebrew/Caskroom/miniforge/base/envs/du for local tests
- AVOID HARDCODING WHENEVER POSSIBLE!
- the user always copies local changes to remote system, and restarts relevant notebooks
- STOP GUESSING!