# Model Switching Guide

**A Complete Guide to Changing Vision-Language Models Without Programming**

Version: 1.0
Last Updated: 2025-11-18

---

## Table of Contents

1. [What is Model Switching?](#what-is-model-switching)
2. [Available Models](#available-models)
3. [Quick Start](#quick-start)
4. [Model Comparison](#model-comparison)
5. [When to Use Which Model](#when-to-use-which-model)
6. [Step-by-Step Switching Instructions](#step-by-step-switching-instructions)
7. [Troubleshooting](#troubleshooting)

---

## What is Model Switching?

Model switching allows you to **change which AI model processes your documents** by editing a single line in your notebook. Different models have different strengths, speeds, and resource requirements.

### Why Switch Models?

✅ **Speed vs. Accuracy Trade-offs**
- Fast models for high-volume processing
- Accurate models for critical documents

✅ **Hardware Constraints**
- Small models for limited GPU memory
- Large models when you have powerful hardware

✅ **Document Type Optimization**
- Some models excel at specific document types
- Test different models to find the best fit

✅ **Cost Management**
- Smaller models = lower GPU costs
- Larger models = higher accuracy but more expensive

---

## Available Models

### Llama-3.2-Vision Models

#### 1. `llama-3.2-11b-vision` (Recommended for Quality)

```yaml
Model: Llama-3.2-11B-Vision-Instruct
Size: 11 billion parameters
Memory: ~22GB GPU memory
Quantization: None (full precision)
Speed: Moderate (2-3 images/minute)
Accuracy: Highest
```

**Best For:**
- Maximum accuracy requirements
- Complex documents with dense information
- When you have sufficient GPU memory (≥24GB)
- Financial documents requiring precision

**Example Use Case:**
> "We process 500 invoices per month and accuracy is critical for accounting. We use llama-3.2-11b-vision despite slower speed."

---

#### 2. `llama-3.2-11b-vision-8bit` (Balanced Option)

```yaml
Model: Llama-3.2-11B-Vision-Instruct
Size: 11 billion parameters
Memory: ~12GB GPU memory (8-bit quantization)
Quantization: 8-bit
Speed: Moderate-Fast (3-4 images/minute)
Accuracy: Very High (95% of full precision)
```

**Best For:**
- Good accuracy with lower memory requirements
- GPUs with 16GB VRAM
- Balanced speed and quality
- Most production use cases

**Example Use Case:**
> "We have V100 GPUs (16GB) and need good accuracy. The 8-bit version gives us 95% of the quality with half the memory."

---

### InternVL3 Models

#### 3. `internvl3-2b` (Fastest Option)

```yaml
Model: InternVL3-2B
Size: 2 billion parameters
Memory: ~4GB GPU memory
Quantization: None (full precision)
Speed: Very Fast (8-10 images/minute)
Accuracy: Good (70-75%)
```

**Best For:**
- High-volume processing (thousands of documents)
- Quick preliminary extraction
- Limited GPU resources
- Development and testing

**Example Use Case:**
> "We process 10,000 receipts daily. We use internvl3-2b for initial extraction, then human review for critical fields."

---

#### 4. `internvl3-8b` (Best Overall Balance)

```yaml
Model: InternVL3-8B
Size: 8 billion parameters
Memory: ~16GB GPU memory
Quantization: None (full precision)
Speed: Fast (5-6 images/minute)
Accuracy: Very High (74%+)
```

**Best For:**
- Production workloads
- Good balance of speed and accuracy
- Modern GPUs (H100, H200, A100, L40)
- General-purpose document extraction

**Example Use Case:**
> "We tested all models on our invoice dataset. InternVL3-8B gave us 74% accuracy at 3x the speed of Llama. Best ROI for our use case."

---

#### 5. `internvl3-8b-quantized` (For Older GPUs)

```yaml
Model: InternVL3-8B
Size: 8 billion parameters
Memory: ~8GB GPU memory (8-bit quantization)
Quantization: 8-bit
Speed: Fast (4-5 images/minute)
Accuracy: High (70%+)
```

**Best For:**
- V100 GPUs or older hardware
- Memory-constrained environments
- When internvl3-8b runs out of memory
- Cloud cost optimization

**Example Use Case:**
> "Our production environment uses V100 GPUs. The quantized 8B model fits perfectly and still delivers 70%+ accuracy."

---

#### 6. `internvl3_5-8b` (Latest Technology)

```yaml
Model: InternVL3.5-8B
Size: 8 billion parameters
Memory: ~16GB GPU memory
Quantization: None (full precision)
Speed: Fast (5-6 images/minute)
Accuracy: Very High (latest improvements)
```

**Best For:**
- Cutting-edge performance
- Taking advantage of latest research
- When you want the newest model capabilities
- Testing new features

**Example Use Case:**
> "We always test the latest models. InternVL3.5 showed 2-3% accuracy improvement over 3.0 on our bank statements."

---

## Quick Start

### Step 1: Open Your Notebook

Open `information_extractor.ipynb` in Jupyter.

### Step 2: Find the Model Configuration Cell

Look for the cell that contains:

```python
CONFIG = {
    'MODEL_NAME': 'internvl3-8b',  # ← This is what you change!
    ...
}
```

### Step 3: Change the Model Name

Replace the model name with your choice:

```python
CONFIG = {
    'MODEL_NAME': 'llama-3.2-11b-vision',  # Changed to Llama!
    ...
}
```

### Step 4: Restart and Run

1. **Restart Kernel:** `Kernel → Restart Kernel`
2. **Run All Cells:** `Cell → Run All`

That's it! The notebook will now use the new model.

---

## Model Comparison

### Performance Matrix

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| **llama-3.2-11b-vision** | ⭐⭐ | ⭐⭐⭐⭐⭐ | 22GB | Maximum accuracy |
| **llama-3.2-11b-vision-8bit** | ⭐⭐⭐ | ⭐⭐⭐⭐ | 12GB | V100 production |
| **internvl3-2b** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 4GB | High volume |
| **internvl3-8b** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 16GB | Best balance |
| **internvl3-8b-quantized** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 8GB | Memory constrained |
| **internvl3_5-8b** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 16GB | Latest features |

### Accuracy Benchmarks (Sample Dataset)

Based on 9-document test set:

```
llama-3.2-11b-vision:      84% average accuracy
llama-3.2-11b-vision-8bit: 82% average accuracy
internvl3-2b:              72% average accuracy
internvl3-8b:              74% average accuracy
internvl3-8b-quantized:    70% average accuracy
internvl3_5-8b:            76% average accuracy
```

**Note:** Your results may vary based on document types and quality.

---

## When to Use Which Model

### Scenario-Based Selection

#### Scenario 1: Financial Audit (Accuracy Critical)

**Requirement:**
- Maximum accuracy
- 200 invoices to process
- Have powerful GPU (A100/H100/H200)

**Recommended Model:** `llama-3.2-11b-vision`

**Configuration:**
```python
CONFIG = {
    'MODEL_NAME': 'llama-3.2-11b-vision',
    'SYSTEM_MODE': 'strict',  # Highest accuracy mode
    ...
}
```

**Why:**
- Highest accuracy (84%)
- Strict mode minimizes errors
- Volume is manageable despite slower speed

---

#### Scenario 2: Daily Receipt Processing (High Volume)

**Requirement:**
- Process 5,000 receipts daily
- Need quick turnaround (4 hours max)
- Accuracy ≥70% acceptable (human review backup)

**Recommended Model:** `internvl3-2b`

**Configuration:**
```python
CONFIG = {
    'MODEL_NAME': 'internvl3-2b',
    'SYSTEM_MODE': 'flexible',  # Faster extraction
    'MAX_IMAGES': None,  # Process all
}
```

**Why:**
- Fastest processing (8-10 images/min = ~8-10 hours for 5k)
- 72% accuracy still useful for bulk processing
- Low memory requirements allow multiple instances

---

#### Scenario 3: Production Deployment (Balanced)

**Requirement:**
- Process 1,000 documents daily
- Need good accuracy (≥74%)
- Cost-effective GPU usage

**Recommended Model:** `internvl3-8b`

**Configuration:**
```python
CONFIG = {
    'MODEL_NAME': 'internvl3-8b',
    'SYSTEM_MODE': 'expert',  # Balanced mode
    ...
}
```

**Why:**
- Excellent balance (74% accuracy, 5-6 images/min)
- Processes 1,000 docs in ~3 hours
- Fits on standard cloud GPUs (A100, L40)

---

#### Scenario 4: Legacy Hardware (V100 GPUs)

**Requirement:**
- Stuck with V100 GPUs (16GB VRAM)
- Need best possible accuracy within constraints
- 500 documents daily

**Recommended Model:** `llama-3.2-11b-vision-8bit`

**Configuration:**
```python
CONFIG = {
    'MODEL_NAME': 'llama-3.2-11b-vision-8bit',
    'SYSTEM_MODE': 'expert',
    ...
}
```

**Why:**
- Best accuracy for V100 hardware (82%)
- Fits in 12GB VRAM (comfortable on 16GB V100)
- Still processes 500 docs in ~2-3 hours

**Alternative:** `internvl3-8b-quantized` if you need more speed

---

#### Scenario 5: Document Type Testing

**Requirement:**
- Testing extraction on new document types
- Want quick feedback
- Don't need perfect accuracy yet

**Recommended Model:** `internvl3-2b`

**Configuration:**
```python
CONFIG = {
    'MODEL_NAME': 'internvl3-2b',
    'MAX_IMAGES': 10,  # Small test set
    'VERBOSE': True,   # See detailed output
}
```

**Why:**
- Fastest iteration cycle
- Quick results for prompt tuning
- Low cost for experimentation

---

## Step-by-Step Switching Instructions

### Complete Example: Switching from InternVL3-8B to Llama

**Starting Point:**
```python
# Current configuration in notebook
CONFIG = {
    'MODEL_NAME': 'internvl3-8b',
    'SYSTEM_MODE': 'expert',
    'DATA_DIR': '/path/to/data',
    ...
}
```

**Step 1: Save Current Results (Optional but Recommended)**

Before switching, note your current performance:
```python
# After running current model, note:
# - Average accuracy: 74%
# - Processing time: 2.1 minutes per image
# - Memory usage: 16GB
```

**Step 2: Change Model Name**

```python
CONFIG = {
    'MODEL_NAME': 'llama-3.2-11b-vision',  # ← Changed!
    'SYSTEM_MODE': 'expert',  # Keep same for fair comparison
    'DATA_DIR': '/path/to/data',  # No change
    ...
}
```

**Step 3: Restart Kernel**

In Jupyter:
- Click: `Kernel → Restart Kernel`
- Confirm the restart

**Step 4: Run All Cells**

- Click: `Cell → Run All`
- Wait for model loading (first run takes 1-2 minutes)

**Step 5: Compare Results**

Look at the output:
```
Before (internvl3-8b):
  Average Accuracy: 74%
  Processing Time: 2.1 min/image
  Memory: 16GB

After (llama-3.2-11b-vision):
  Average Accuracy: 84%  ← 10% improvement!
  Processing Time: 3.5 min/image  ← Slower
  Memory: 22GB  ← More memory
```

**Step 6: Decide**

Questions to ask:
- Is the accuracy improvement worth the slower speed?
- Do I have enough GPU memory (22GB)?
- Is the processing time acceptable for my volume?

---

### Example: Switching Between InternVL3 Versions

**Testing Different InternVL3 Sizes:**

```python
# Test 1: Small and fast
CONFIG = {
    'MODEL_NAME': 'internvl3-2b',
    'MAX_IMAGES': 100,  # Test subset
}
# Run → Note: 72% accuracy, 1.2 min/image

# Test 2: Larger and more accurate
CONFIG = {
    'MODEL_NAME': 'internvl3-8b',
    'MAX_IMAGES': 100,  # Same subset
}
# Run → Note: 74% accuracy, 2.1 min/image

# Test 3: Latest version
CONFIG = {
    'MODEL_NAME': 'internvl3_5-8b',
    'MAX_IMAGES': 100,  # Same subset
}
# Run → Note: 76% accuracy, 2.2 min/image

# Decision: internvl3_5-8b gives best accuracy with minimal speed penalty
CONFIG = {
    'MODEL_NAME': 'internvl3_5-8b',  # Final choice
    'MAX_IMAGES': None,  # Process everything
}
```

---

## Troubleshooting

### Problem: Model Won't Load

**Symptoms:**
```
Error: Model path not found: /path/to/model
```

**Solution:**

1. **Check model path configuration:**
```yaml
# In config/models.yaml
model_paths:
  llama:
    production: "/home/user/models/Llama-3.2-11B-Vision-Instruct"
  internvl3:
    production: "/home/user/models/InternVL3-8B"
```

2. **Verify files exist:**
```bash
ls /home/user/models/Llama-3.2-11B-Vision-Instruct
ls /home/user/models/InternVL3-8B
```

3. **Download if missing:**
```bash
# Llama (requires authentication)
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-11B-Vision-Instruct

# InternVL3 (public)
huggingface-cli download OpenGVLab/InternVL3-8B
```

---

### Problem: Out of Memory Error

**Symptoms:**
```
CUDA out of memory. Tried to allocate 22.00 GB
```

**Solutions:**

**Option 1: Use quantized version**
```python
# If using:
CONFIG = {'MODEL_NAME': 'llama-3.2-11b-vision'}  # 22GB

# Switch to:
CONFIG = {'MODEL_NAME': 'llama-3.2-11b-vision-8bit'}  # 12GB
```

**Option 2: Use smaller model**
```python
# Switch to smaller model entirely
CONFIG = {'MODEL_NAME': 'internvl3-2b'}  # 4GB
```

**Option 3: Clear GPU memory first**
```python
# Run cleanup cell before model loading
from common.gpu_optimization import emergency_cleanup
emergency_cleanup(verbose=True)
```

---

### Problem: Accuracy Dropped After Switching

**Symptoms:**
```
Previous model: 84% accuracy
New model: 72% accuracy
```

**Investigation Steps:**

1. **Check which fields are failing:**
```python
# Look at field-level accuracy in output CSV
df = pd.read_csv('output/csv/results.csv')
for field in FIELD_COLUMNS:
    accuracy = df[f'{field}_accuracy'].mean()
    print(f"{field}: {accuracy:.1%}")
```

2. **Try different system mode:**
```python
# Current (flexible mode may be too loose)
CONFIG = {'SYSTEM_MODE': 'flexible'}

# Try stricter mode
CONFIG = {'SYSTEM_MODE': 'expert'}  # or 'precise'
```

3. **Adjust for specific document types:**
```python
# Some models are better at specific types
# Test on document type subsets
CONFIG = {'DOCUMENT_TYPES': ['invoice']}  # Test invoices only
```

---

### Problem: Processing Too Slow

**Symptoms:**
```
Processing 1,000 images taking 60+ hours
```

**Solutions:**

**Immediate: Switch to faster model**
```python
CONFIG = {
    'MODEL_NAME': 'internvl3-2b',  # 3-4x faster
}
```

**Optimize: Reduce preprocessing**
```python
CONFIG = {
    'ENABLE_PREPROCESSING': False,  # Skip image cleanup
    'ENABLE_FIXING': False,         # Skip self-healing
}
```

**Scale: Run parallel instances**
```python
# Split dataset into chunks
# Run multiple notebooks simultaneously
# Example: 4 notebooks processing 250 images each
```

---

### Problem: Model Giving Wrong Document Type

**Symptoms:**
```
Invoice detected as: RECEIPT
Receipt detected as: INVOICE
```

**Solutions:**

1. **Check if model-specific issue:**
```python
# Try different model
CONFIG = {'MODEL_NAME': 'llama-3.2-11b-vision'}
# Llama often better at document type detection
```

2. **Enhance detection prompt:**
```yaml
# In config/prompts.yaml
detection_prompt: |
  Classify this document carefully.

  INVOICE = Has "Invoice Number", "Amount Due", "Payment Terms"
  RECEIPT = Has "Paid", "Transaction Complete", "Thank You"

  Respond with only: DOCUMENT_TYPE: [type]
```

---

## Model Selection Decision Tree

```
START: What's most important?

├─ ACCURACY (financial, legal, critical)
│  ├─ Have powerful GPU (A100/H100/H200)?
│  │  └─ YES → llama-3.2-11b-vision
│  └─ Limited to V100 or lower?
│     └─ YES → llama-3.2-11b-vision-8bit
│
├─ SPEED (high volume, time-sensitive)
│  ├─ Need ≥70% accuracy?
│  │  └─ YES → internvl3-8b
│  └─ 65-70% accuracy acceptable?
│     └─ YES → internvl3-2b
│
├─ COST (budget constrained)
│  ├─ Can accept 70% accuracy?
│  │  └─ YES → internvl3-2b (cheapest)
│  └─ Need 74%+ accuracy?
│     └─ YES → internvl3-8b-quantized
│
└─ BALANCED (production deployment)
   └─ internvl3-8b or internvl3_5-8b
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│              MODEL SWITCHING QUICK GUIDE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HOW TO SWITCH:                                             │
│                                                             │
│  1. Open notebook                                           │
│  2. Find: CONFIG = {'MODEL_NAME': '...'}                    │
│  3. Change model name                                       │
│  4. Kernel → Restart Kernel                                 │
│  5. Cell → Run All                                          │
│                                                             │
│  AVAILABLE MODELS:                                          │
│                                                             │
│  Fast & Cheap:                                              │
│    internvl3-2b                   (4GB, 72%, 8-10 img/min)  │
│                                                             │
│  Balanced:                                                  │
│    internvl3-8b                   (16GB, 74%, 5-6 img/min)  │
│    internvl3_5-8b                 (16GB, 76%, 5-6 img/min)  │
│                                                             │
│  High Accuracy:                                             │
│    llama-3.2-11b-vision-8bit      (12GB, 82%, 3-4 img/min)  │
│    llama-3.2-11b-vision           (22GB, 84%, 2-3 img/min)  │
│                                                             │
│  Memory Constrained:                                        │
│    internvl3-8b-quantized         (8GB, 70%, 4-5 img/min)   │
│                                                             │
│  QUICK DECISION:                                            │
│                                                             │
│    Need maximum accuracy?  → llama-3.2-11b-vision           │
│    Have V100 GPU?         → llama-3.2-11b-vision-8bit       │
│    Processing 1000s/day?  → internvl3-2b                    │
│    Best overall balance?  → internvl3-8b                    │
│    Latest & greatest?     → internvl3_5-8b                  │
│                                                             │
│  TROUBLESHOOTING:                                           │
│                                                             │
│    Out of memory?         → Use -8bit or -quantized version │
│    Too slow?              → Use smaller model (2b or 8b)    │
│    Accuracy too low?      → Use larger model (11b)          │
│    Wrong doc type?        → Try llama (better detection)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Conclusion

Model switching gives you powerful flexibility to optimize for your specific needs:

✅ **Test multiple models** on your documents
✅ **Measure** accuracy, speed, and cost trade-offs
✅ **Choose** based on your priorities
✅ **Adjust** as your requirements change

Remember:
- **No single "best" model** - it depends on your use case
- **Test with your actual documents** - benchmarks are guides, not guarantees
- **Start with balanced model** (internvl3-8b) then optimize
- **Document your findings** to inform future decisions

---

**Document Version:** 1.0
**Last Updated:** 2025-11-18
**Configuration File:** `config/models.yaml`
**Notebook:** `information_extractor.ipynb`
