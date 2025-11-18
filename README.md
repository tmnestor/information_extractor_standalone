# Information Extractor - LangChain Vision-Language Pipeline

**Production-ready document extraction achieving 84.12% accuracy using vision-language models (Llama-3.2-Vision, InternVL3) with LangChain integration.**

## 🎯 Overview

This standalone notebook provides a complete pipeline for extracting structured data from business documents (invoices, receipts, bank statements) using state-of-the-art vision-language models. The system features:

- **High Accuracy**: 84.12% average (100% on some documents!)
- **LangChain v1.0 Integration**: Full BaseChatModel compatibility
- **Hot-Reload Configuration**: Edit prompts/configs without restarting
- **Multi-Model Support**: Switch between Llama-3.2-Vision and InternVL3
- **Production-Ready**: Comprehensive analytics, visualizations, and reporting
- **Type-Safe**: Pydantic v2 schemas with automatic validation

## ✨ Key Features

### 1. **LangChain Pipeline** (`information_extractor.ipynb`)
- Document type detection (invoice/receipt/bank_statement)
- Field extraction with Pydantic validation
- Production-ready cleaning system (CRITICAL for accuracy)
- Batch processing with progress tracking
- Comprehensive evaluation against ground truth

### 2. **YAML-Based Configuration**
- `config/models.yaml` - Model configurations (paths, quantization, parameters)
- `config/prompts.yaml` - Extraction prompts and system messages
- `config/field_definitions.yaml` - Document schemas and field definitions

### 3. **Accuracy Optimization**
- **ExtractionCleaner**: Normalizes LLM output to ground truth format
- **Serialization Pipeline**: Converts Pydantic types → LLM strings → Cleaned output
- **Field-Level Metrics**: Track accuracy per field, per document type
- **Validation Tests**: 82%+ baseline from `test_langchain_integration.py`

## 📋 Requirements

### System Requirements
- **Python**: 3.11+
- **RAM**: 16GB+ system memory
- **GPU**: CUDA-capable GPU (optional but recommended)
  - Llama-3.2-11B: 16GB+ VRAM (or 8-bit quantization for 8GB)
  - InternVL3-2B: 4GB+ VRAM (memory efficient)
  - InternVL3-8B: 16GB+ VRAM (or quantization for 8GB)
- **Storage**: ~30GB for models + data

### Software Dependencies
All dependencies are in `environment.yml`:
- **Core**: PyTorch, transformers, accelerate, bitsandbytes
- **LangChain**: langchain-core, langchain
- **Data**: pandas, numpy, Pillow
- **Validation**: pydantic (v2)
- **Visualization**: matplotlib, seaborn, rich

## 🚀 Quick Start

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate information_extractor
```

### 2. Download Models

**Option A: Automatic Download (via Hugging Face Hub)**
```python
# Models will be downloaded automatically on first run
# Just set model_id in config/models.yaml
```

**Option B: Manual Download**
```bash
# Download Llama-3.2-Vision
git clone https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

# Download InternVL3
git clone https://huggingface.co/OpenGVLab/InternVL3-2B
# or
git clone https://huggingface.co/OpenGVLab/InternVL3-8B
```

### 3. Update Configuration

**A. Update Model Paths** in `config/models.yaml`:
```yaml
model_paths:
  llama:
    production: "/path/to/Llama-3.2-11B-Vision-Instruct"  # UPDATE THIS
  internvl3:
    production: "/path/to/InternVL3-8B"  # UPDATE THIS
```

**B. Update Data Paths** in `information_extractor.ipynb` (Cell-6 CONFIG):
```python
CONFIG = {
    'MODEL_NAME': 'llama-3.2-11b-vision-8bit',  # or 'internvl3-2b', 'internvl3-8b'
    'DATA_DIR': '/path/to/evaluation_data',  # UPDATE THIS
    'GROUND_TRUTH': '/path/to/evaluation_data/ground_truth.csv',  # UPDATE THIS
    'OUTPUT_BASE': './output',
}
```

### 4. Run the Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: information_extractor.ipynb
# Select kernel: "information_extractor" (or your env name)
# Run All Cells
```

### 5. Test with Sample Data

The repository includes sample data for validation:
- `evaluation_data/ground_truth.csv` - 3 sample documents (1 invoice, 1 receipt, 1 bank statement)
- `evaluation_data/images/sample_*.png` - Corresponding sample images

**To test:**
1. Set `DATA_DIR` to `./evaluation_data`
2. Set `GROUND_TRUTH` to `./evaluation_data/ground_truth.csv`
3. Run notebook - should achieve ~80%+ accuracy on samples

## 📖 Usage Guide

### Running Inference (No Ground Truth)

```python
# In Cell-6 CONFIG:
CONFIG = {
    'MODEL_NAME': 'llama-3.2-11b-vision-8bit',
    'DATA_DIR': '/path/to/your/images',
    'INFERENCE_ONLY': True,  # Set to True
    'GROUND_TRUTH': None,    # Not needed
}

# Run All Cells
# Results in: output/csv/llama_batch_results_{timestamp}.csv
```

### Running Evaluation (With Ground Truth)

```python
# In Cell-6 CONFIG:
CONFIG = {
    'MODEL_NAME': 'llama-3.2-11b-vision-8bit',
    'DATA_DIR': '/path/to/evaluation_images',
    'GROUND_TRUTH': '/path/to/ground_truth.csv',
    'INFERENCE_ONLY': False,  # Set to False
}

# Run All Cells
# Results include accuracy metrics, field-level analysis, visualizations
```

### Switching Models

**Edit Cell-6 CONFIG:**
```python
# For Llama-3.2-Vision (best quality, 16GB VRAM)
'MODEL_NAME': 'llama-3.2-11b-vision'

# For Llama-3.2-Vision 8-bit (memory efficient, 8GB VRAM)
'MODEL_NAME': 'llama-3.2-11b-vision-8bit'

# For InternVL3-2B (lightweight, 4GB VRAM)
'MODEL_NAME': 'internvl3-2b'

# For InternVL3-8B (strong performance, 16GB VRAM)
'MODEL_NAME': 'internvl3-8b'

# For InternVL3-8B quantized (V100 compatible)
'MODEL_NAME': 'internvl3-8b-quantized'
```

### Hot-Reload Prompts

Edit `config/prompts.yaml`, then in notebook:
```python
prompt_manager.reload_config()
# Re-run processing cells to use new prompts
```

## 🏗️ Architecture

### Pipeline Stages

```
1. Document Type Detection
   ↓
2. Field Extraction (Pydantic validation)
   ↓
3. **Serialization** (Pydantic → LLM string format)
   ↓
4. **Cleaning** (ExtractionCleaner normalization)
   ↓
5. Evaluation (if ground truth available)
   ↓
6. Analytics & Reporting
```

### Critical Components

**1. Serialization (`serialize_pydantic_to_llm_format`)**
- Converts Pydantic types to LLM string format
- Lists → `"item1, item2"`
- Decimals → `"$123.45"`
- Bools → `"true"`/`"false"`

**2. Cleaning (`ExtractionCleaner`)**
- Normalizes LLM output to ground truth format
- ABN formatting: `06082698025` → `06 082 698 025`
- List conversion: `"item1, item2"` → `"item1 | item2"`
- Address cleaning, date parsing, type normalization

**3. Evaluation (`calculate_field_accuracy_with_method`)**
- Order-aware F1 scoring for lists
- Fuzzy matching for addresses
- Exact matching for IDs
- Field-level and document-level metrics

## 📊 Performance

### Accuracy Metrics (9 test documents)

| Metric | Value |
|--------|-------|
| **Average Accuracy** | 84.12% |
| **Median Accuracy** | 92.86% |
| **Range** | 51.5% - 100% |
| **Perfect Scores** | 1 (image_002.png) |

### By Document Type

| Document Type | Accuracy | Sample Size |
|---------------|----------|-------------|
| Receipts | ~93% avg | 3 documents |
| Invoices | ~92% avg | 3 documents |
| Bank Statements | ~73% avg | 3 documents |

### Processing Speed
- **Average**: 11.6 seconds per image
- **Throughput**: 5.2 images/minute
- **Tested on**: H200 GPU with 8-bit quantization

## 🧪 Testing

### Run Validation Tests

```bash
# Activate environment
conda activate information_extractor

# Run integration test (validates 82%+ accuracy baseline)
python tests/test_langchain_integration.py
```

**Expected output:**
```
✅ Test passed: 9/9 documents processed
✅ Average accuracy: 84.1% (target: 82%)
✅ All document types validated
```

## 📁 Project Structure

```
information_extractor_standalone/
├── information_extractor.ipynb     # Main processing notebook
├── README.md                       # This file
├── .gitignore                      # Git ignore patterns
├── environment.yml                 # Conda dependencies
│
├── common/                         # Core Python modules (21 files)
│   ├── langchain_chains.py         # LangChain pipeline
│   ├── langchain_llm.py            # Vision-language model wrapper
│   ├── langchain_prompts.py        # YAML prompt management
│   ├── extraction_cleaner.py       # CRITICAL for accuracy
│   ├── extraction_schemas.py       # Pydantic validation
│   ├── evaluation_metrics.py       # Accuracy calculation
│   └── ... (15 more modules)
│
├── config/                         # YAML configurations
│   ├── models.yaml                 # Model settings (UPDATE PATHS)
│   ├── prompts.yaml                # Extraction prompts
│   └── field_definitions.yaml      # Document schemas
│
├── tests/                          # Validation tests
│   └── test_langchain_integration.py
│
├── docs/                           # Documentation
│   ├── CLAUDE.md                   # Development guidelines
│   └── SERIALIZATION_FIX_SUMMARY.md # Technical fix docs
│
└── evaluation_data/                # Sample test data
    ├── ground_truth.csv            # 3 sample documents
    └── images/
        ├── sample_image_001.png    # Sample receipt
        ├── sample_image_003.png    # Sample bank statement
        └── sample_image_005.png    # Sample invoice
```

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Solutions:**
1. Use 8-bit quantization: `MODEL_NAME: 'llama-3.2-11b-vision-8bit'`
2. Reduce batch size: Process images one at a time
3. Use smaller model: `MODEL_NAME: 'internvl3-2b'`
4. Clear GPU memory: Run Cell-4 (emergency cleanup)

### Issue: Low Accuracy (<80%)

**Checklist:**
1. ✅ Verify ExtractionCleaner is enabled (Cell-18)
2. ✅ Check serialization step is present (Cell-18: `serialize_pydantic_to_llm_format`)
3. ✅ Verify ground truth format matches (pipe-separated lists: `"item1 | item2"`)
4. ✅ Run validation test: `python tests/test_langchain_integration.py`

### Issue: ModuleNotFoundError

**Solution:**
```bash
# Ensure environment is activated
conda activate information_extractor

# Verify all packages installed
conda list | grep -E "transformers|langchain|pydantic"

# Reinstall if needed
conda env update -f environment.yml --prune
```

### Issue: Model Not Found

**Solutions:**
1. Check `config/models.yaml` paths are absolute
2. Verify model directory exists: `ls /path/to/model`
3. Try Hugging Face auto-download: Use `model_id: "meta-llama/Llama-3.2-11B-Vision-Instruct"` directly

## 📚 Additional Documentation

- **Development Guidelines**: `docs/CLAUDE.md`
- **Technical Fix Details**: `docs/SERIALIZATION_FIX_SUMMARY.md`
- **LangChain Integration**: Cell comments in notebook
- **Prompt Engineering**: `config/prompts.yaml` (with inline comments)

## 🎓 Key Learnings

### The Serialization Fix (CRITICAL)

**Problem:** ExtractionCleaner expects LLM string output, not Python typed objects.

**Before (60% accuracy):**
```python
# Pydantic model outputs:
LINE_ITEM_DESCRIPTIONS = ['Energy Drink', 'Premium Unleaded']  # Python List
TOTAL_AMOUNT = Decimal('57.15')                                 # Python Decimal

# ❌ Passed directly to cleaner → type mismatch → low accuracy
```

**After (84% accuracy):**
```python
# Serialization step:
LINE_ITEM_DESCRIPTIONS = "Energy Drink, Premium Unleaded"       # LLM string
TOTAL_AMOUNT = "$57.15"                                         # LLM string

# ✅ Cleaner converts: "item1, item2" → "item1 | item2" → matches ground truth!
```

**Implementation:** See Cell-18 `serialize_pydantic_to_llm_format()` function

## 🤝 Contributing

This is a standalone repository for production use. For enhancements:
1. Test changes thoroughly with `test_langchain_integration.py`
2. Verify accuracy remains ≥82%
3. Update documentation as needed

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- **LangChain Team** - BaseChatModel framework
- **Meta AI** - Llama-3.2-Vision model
- **OpenGVLab** - InternVL3 models
- **Hugging Face** - Transformers library

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review documentation in `docs/`
3. Run validation tests to verify setup
4. Check Jupyter notebook cell comments for inline guidance

---

**Happy Extracting!** 🚀

**Current Version**: 84.12% accuracy baseline (validated 2025-11-18)
