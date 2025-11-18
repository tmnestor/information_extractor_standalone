# Prompt Configuration Guide

**A Complete Guide to Customizing Document Extraction Without Writing Code**

Version: 1.0
Last Updated: 2025-11-18

---

## Table of Contents

1. [What is Prompt Configuration?](#what-is-prompt-configuration)
2. [Quick Start](#quick-start)
3. [Understanding the Configuration File](#understanding-the-configuration-file)
4. [System Prompt Modes](#system-prompt-modes)
5. [Customizing Document Instructions](#customizing-document-instructions)
6. [Extraction Rules](#extraction-rules)
7. [Real-World Examples](#real-world-examples)
8. [Advanced Configuration](#advanced-configuration)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## What is Prompt Configuration?

Prompt configuration allows you to **control how the AI model extracts information from documents** by editing a simple YAML text file. No programming required!

### Why This Matters

- ✅ **Change extraction behavior** without touching code
- ✅ **Test different extraction strategies** instantly
- ✅ **Customize for your specific documents** and requirements
- ✅ **Switch between modes** (strict vs. flexible, etc.)
- ✅ **Hot-reload changes** while the system is running

### What You Can Control

- How the AI "thinks" about documents (expert, precise, flexible, etc.)
- Specific instructions for different document types
- Formatting rules for extracted data
- Field naming and organization
- Error handling and edge cases

---

## Quick Start

### Step 1: Find the Configuration File

The main prompt configuration file is located at:
```
config/prompts.yaml
```

### Step 2: Open and Edit

Open `config/prompts.yaml` in any text editor (Notepad, VS Code, etc.).

### Step 3: Make Changes

Edit the file following the examples below, then save.

### Step 4: Reload (Optional)

If your notebook supports hot-reload, changes take effect immediately. Otherwise, restart the notebook from the model loading cell.

---

## Understanding the Configuration File

The `prompts.yaml` file is organized into sections:

```yaml
# Main sections:
system_prompts:      # How the AI "thinks"
document_instructions: # Specific guidance per document type
extraction_rules:    # Formatting and data rules
conversation_protocol: # Response format requirements
```

### File Structure Example

```yaml
system_prompts:
  expert: "You are an expert document analyzer..."
  precise: "Extract only clearly visible information..."

document_instructions:
  invoice: "This is an INVOICE. Focus on vendor info..."
  receipt: "This is a RECEIPT. Focus on merchant..."

extraction_rules: |
  - Use exact text from document
  - Be conservative with NOT_FOUND
  - Include currency symbols
```

---

## System Prompt Modes

System prompts define the AI's "personality" and approach to extraction.

### Available Modes

#### 1. **expert** (Default) - Recommended for Most Use Cases

```yaml
expert: "You are an expert document analyzer specialized in {document_type} extraction."
```

**When to use:**
- General-purpose document extraction
- Balanced accuracy and completeness
- Standard business documents

**Example behavior:**
- Extracts visible information confidently
- Makes reasonable inferences for standard fields
- Handles common document variations well

---

#### 2. **precise** - For High-Accuracy Requirements

```yaml
precise: "Extract only clearly visible information. Use NOT_FOUND for any unclear fields."
```

**When to use:**
- Financial audits requiring certainty
- Legal documents where accuracy is critical
- When you prefer missing data over incorrect data

**Example behavior:**
- Only extracts 100% clear fields
- Uses "NOT_FOUND" liberally
- Won't guess or infer missing information

**Example output:**
```
SUPPLIER_NAME: Acme Corp
INVOICE_NUMBER: NOT_FOUND  # Partially obscured
TOTAL_AMOUNT: $1,234.56
```

---

#### 3. **flexible** - For Challenging Documents

```yaml
flexible: "Extract information, making reasonable inferences for partially visible fields."
```

**When to use:**
- Poor quality scans
- Partially damaged documents
- When some data is better than no data

**Example behavior:**
- Attempts extraction even with partial information
- Makes educated guesses based on context
- More complete results, but slightly higher error rate

**Example output:**
```
SUPPLIER_NAME: Acme Corp  # Partially visible, inferred from logo
INVOICE_NUMBER: INV-2023-... # Partially obscured
TOTAL_AMOUNT: $1,234.56
```

---

#### 4. **strict** - For Legal/Compliance

```yaml
strict: "Extract ONLY information that is completely clear and unambiguous."
```

**When to use:**
- Legal discovery
- Regulatory compliance
- When false positives are unacceptable

**Example behavior:**
- Highest accuracy standard
- Maximum use of "NOT_FOUND"
- Zero tolerance for uncertainty

---

#### 5. **structured** - For Standardized Documents

```yaml
structured: "You are a structured data extraction system. Extract information with perfect formatting."
```

**When to use:**
- Standardized forms with consistent layouts
- Machine-generated documents
- When format consistency is critical

**Example behavior:**
- Enforces strict formatting rules
- Expects consistent field placement
- Optimized for template-based documents

---

### Changing the Active Mode

**In `config/prompts.yaml`:**
```yaml
default_system_mode: "expert"  # Change this to switch modes
```

**In your notebook:**
```python
CONFIG = {
    'SYSTEM_MODE': 'precise',  # Override in notebook
    ...
}
```

---

## Customizing Document Instructions

Document instructions provide specific guidance for each document type.

### Invoice Instructions

**Current default:**
```yaml
invoice: |
  This is an INVOICE document. Focus on:
  - Vendor/supplier information (name, ABN, address)
  - Customer/payer information
  - Line items with quantities and prices
  - GST and total amounts
  - Invoice date and number
```

**Example customization for your industry:**

```yaml
invoice: |
  This is a CONSTRUCTION INVOICE. Focus on:
  - Contractor details (license number required)
  - Project name and address
  - Labor and materials breakdown
  - Daily rates and hours worked
  - GST, retainage, and final total
  - Progress billing percentage if applicable
```

---

### Receipt Instructions

**Current default:**
```yaml
receipt: |
  This is a RECEIPT document. Focus on:
  - Merchant information
  - Transaction date and time
  - Purchased items with quantities and prices
  - Payment method and totals
  - GST if applicable
```

**Example customization for retail:**

```yaml
receipt: |
  This is a RETAIL RECEIPT. Focus on:
  - Store name and location
  - Transaction ID and cashier number
  - EACH item with SKU, description, quantity, price
  - Discounts and promotions applied
  - Payment method (cash, card last 4 digits)
  - Return policy timeframe
  - Customer loyalty program number if present
```

---

### Bank Statement Instructions

**Current default:**
```yaml
bank_statement: |
  This is a BANK STATEMENT document. Focus on:
  - Statement date range
  - ALL individual transactions (do NOT combine)
  - Transaction descriptions exactly as shown
  - Transaction dates (one per transaction)
  - Amounts paid (debits/withdrawals)
  - CRITICAL: Extract each transaction separately
```

**Example customization for reconciliation:**

```yaml
bank_statement: |
  This is a BANK STATEMENT for reconciliation. Focus on:
  - Account number (last 4 digits)
  - Statement period (opening and closing dates)
  - Opening and closing balances
  - EVERY transaction in chronological order:
    * Date (DD/MM/YYYY format)
    * Description (exact text, no abbreviation)
    * Debit amount (withdrawals/payments)
    * Credit amount (deposits/receipts)
    * Running balance after each transaction
  - Identify check numbers if present
  - Flag any fees or interest charges
```

---

## Extraction Rules

Extraction rules control data formatting and quality standards.

### Current Default Rules

```yaml
extraction_rules: |
  RULES:
  - Use exact text from document
  - CRITICAL: Use ONLY pipe separators (|) for lists
  - Be conservative: use NOT_FOUND if field is truly missing
  - For monetary values: Include currency symbol (e.g., $123.45)
  - For dates: Match format shown in document
  - For lists: Separate items with " | " (space-pipe-space)
  - Do NOT invent or hallucinate information
```

### Customizing Rules

#### Example 1: Strict Date Formatting

```yaml
extraction_rules: |
  RULES:
  - Use exact text from document
  - CRITICAL: Use ONLY pipe separators (|) for lists
  - CRITICAL: Convert ALL dates to DD/MM/YYYY format
    * "Jan 15, 2025" → "15/01/2025"
    * "2025-01-15" → "15/01/2025"
    * "15-Jan-25" → "15/01/2025"
  - For monetary values: Always use $ symbol and 2 decimals
  - For ABN: Always format as XX XXX XXX XXX (with spaces)
  - Use NOT_FOUND only when field is completely absent
```

#### Example 2: Flexible with Partial Data

```yaml
extraction_rules: |
  RULES:
  - Extract visible text, even if incomplete
  - Use pipe separators (|) for lists
  - For partially visible data: Include what's visible + "..."
    * Example: "Acme Corp..." if only partial name visible
  - For monetary values: Include $ symbol when present
  - For dates: Use any clear format (DD/MM/YYYY preferred)
  - Mark uncertain extractions with [?] suffix
  - Use NOT_FOUND only when absolutely nothing is visible
```

#### Example 3: Industry-Specific Rules (Medical)

```yaml
extraction_rules: |
  RULES:
  - Use exact medical terminology from document
  - For medication lists: Drug name | Dosage | Frequency
  - For dates: Always DD/MM/YYYY format
  - For provider names: Title + Full Name (e.g., "Dr. Jane Smith")
  - For diagnosis codes: Include both code and description
  - HIPAA compliance: Do NOT extract patient SSN or full DOB
  - Use NOT_FOUND for redacted information
```

---

## Real-World Examples

### Example 1: Processing Construction Invoices

**Problem:** Default prompts miss important construction-specific fields like license numbers and project codes.

**Solution:**

```yaml
system_prompts:
  construction_expert: "You are an expert in construction billing and project accounting."

document_instructions:
  invoice: |
    This is a CONSTRUCTION INVOICE. Extract:
    - Contractor name and license number
    - Project name and site address
    - Work order or project number
    - Itemized costs (Labor | Materials | Equipment | Subcontractors)
    - Daily/hourly rates with quantities
    - Progress percentage and retainage held
    - Previous payments and current amount due
    - GST and any bond requirements

extraction_rules: |
  RULES:
  - Separate labor and materials clearly
  - Format license numbers as: LIC-XXXXX
  - For project codes: Always include full alphanumeric code
  - For retainage: Show as percentage and dollar amount
  - Extract unit prices AND total prices for each line item
  - Use pipe separators for multiple items
```

**Update notebook:**
```python
CONFIG = {
    'SYSTEM_MODE': 'construction_expert',
    ...
}
```

---

### Example 2: Restaurant Receipts

**Problem:** Missing itemized food orders and tip amounts.

**Solution:**

```yaml
document_instructions:
  receipt: |
    This is a RESTAURANT RECEIPT. Extract:
    - Restaurant name and location
    - Server name and table number
    - Date and time of service
    - EACH menu item ordered (description | quantity | price)
    - Subtotal, tax, and tip amounts separately
    - Total amount charged
    - Payment method (cash, card type, last 4 digits)
    - Any special requests or modifications to orders

extraction_rules: |
  RULES:
  - For menu items: Use pipe format: "Item name | Qty | Unit price | Total"
  - Extract tip as separate field (even if handwritten)
  - For modifications: Include in item description
    * Example: "Burger | 1 | $12.99 | No onions"
  - Time format: HH:MM AM/PM
  - Separate food/beverage/alcohol if itemized separately
```

---

### Example 3: Utility Bills

**Problem:** Need to track usage metrics, not just amounts.

**Solution:**

```yaml
document_instructions:
  utility_bill: |
    This is a UTILITY BILL (electricity/gas/water). Extract:
    - Service provider and account number
    - Billing period (start date | end date)
    - Previous meter reading and current meter reading
    - Total usage (with units: kWh, cubic meters, gallons)
    - Usage charges by tier/rate
    - Fixed/connection fees
    - Taxes and surcharges itemized
    - Total amount due and due date
    - Previous balance if any
    - Late payment penalties and reconnection fees

extraction_rules: |
  RULES:
  - For meter readings: Include units (e.g., "1234 kWh")
  - For tiered rates: Show usage | rate | charge for each tier
  - Dates: DD/MM/YYYY format
  - Separate current charges from previous balance
  - Flag disconnection warnings if present
```

---

### Example 4: Multi-Currency Invoices

**Problem:** International invoices with multiple currencies.

**Solution:**

```yaml
extraction_rules: |
  RULES:
  - For monetary values: ALWAYS include currency code
    * USD amounts: $123.45 USD
    * EUR amounts: €123.45 EUR
    * GBP amounts: £123.45 GBP
  - Extract exchange rate if shown on document
  - Show both foreign currency AND local currency amounts
    * Format: "$100.00 USD ($150.00 AUD at 1.5)"
  - For totals: Clearly mark which currency is primary
  - Use pipe separators for multi-currency line items
```

---

### Example 5: Scanning Historical Documents (OCR Challenges)

**Problem:** Old, degraded documents with poor OCR quality.

**Solution:**

```yaml
system_prompts:
  historical: "You are analyzing historical documents that may have degraded quality. Extract what is clearly visible and mark uncertainties."

extraction_rules: |
  RULES:
  - For partially legible text: Extract visible portion + "[illegible]"
  - For uncertain readings: Add [?] after the field
    * Example: "Smith Bros [?]" if name is unclear
  - For damaged areas: Use "[damaged]" marker
  - For handwritten text: Prefix with [handwritten] if critical
  - Date format flexibility: Accept any clear date format
  - Preserve original spelling, even if appears incorrect
  - Use NOT_FOUND only for completely missing sections

document_instructions:
  invoice: |
    This is a HISTORICAL INVOICE (possibly degraded). Focus on:
    - Date (any format acceptable, mark if uncertain)
    - Vendor/supplier (extract even partial names)
    - Total amount (this is most critical)
    - Any clearly visible line items
    - Signatures or stamps (note presence even if illegible)
    - Document condition: Flag tears, stains, fading
```

---

## Advanced Configuration

### Creating Custom Document Types

You can add entirely new document types:

```yaml
document_instructions:
  purchase_order: |
    This is a PURCHASE ORDER. Extract:
    - PO number and revision
    - Buyer organization and department
    - Vendor details
    - Ship-to and bill-to addresses
    - Requested delivery date
    - Itemized products/services ordered
    - Unit prices and extended totals
    - Terms and conditions
    - Approval signatures

  shipping_manifest: |
    This is a SHIPPING MANIFEST. Extract:
    - Shipment tracking number
    - Carrier name and service level
    - Ship date and expected delivery
    - Origin and destination addresses
    - Package count and weights
    - Itemized contents
    - Declared value and insurance
    - Special handling instructions
```

**Use in notebook:**
```python
# Detection will automatically identify these types
# if they're in the detection prompt
```

---

### Multi-Language Support

```yaml
extraction_rules: |
  RULES (MULTI-LANGUAGE):
  - Extract text in original language
  - For field names: Keep in English
  - For values: Preserve original language
    * SUPPLIER_NAME: "Société Générale"  ✓
    * SUPPLIER_NAME: "General Society"   ✗
  - For currencies: Use local symbols (€, £, ¥)
  - For dates: Preserve local format but note format
    * French: DD/MM/YYYY
    * US: MM/DD/YYYY
  - For special characters: Preserve accents and diacritics
```

---

### Field-Specific Formatting

```yaml
field_format_guidelines:
  # Existing guidelines
  text: "Plain text exactly as shown"
  monetary: "Currency symbol + amount (e.g., $123.45)"
  date: "DD/MM/YYYY or DD-MMM-YY format"
  list: "Item 1 | Item 2 | Item 3 (pipe-separated)"
  boolean: "true or false (lowercase)"
  abn: "XX XXX XXX XXX (with spaces)"

  # Custom guidelines you can add
  phone: "Country code + number (e.g., +61 2 1234 5678)"
  email: "Lowercase, preserve exact format"
  url: "Include https:// protocol"
  percentage: "Number + % symbol (e.g., 10.5%)"
  weight: "Value + unit (e.g., 150.5 kg)"
  dimensions: "L x W x H + units (e.g., 10 x 5 x 3 cm)"
```

---

## Troubleshooting

### Problem: Too Many "NOT_FOUND" Results

**Symptoms:**
- Model extracts very little data
- Many fields marked as NOT_FOUND
- Actual data is visible in documents

**Solutions:**

1. **Switch to flexible mode:**
   ```yaml
   default_system_mode: "flexible"
   ```

2. **Adjust extraction rules:**
   ```yaml
   extraction_rules: |
     - Extract visible information, even if partially obscured
     - Use NOT_FOUND only when field is completely absent
     - For unclear text, extract best interpretation
   ```

3. **Add encouragement in document instructions:**
   ```yaml
   invoice: |
     This is an INVOICE. Extract ALL visible information.
     If a field is partially visible, extract what you can see.
     Only use NOT_FOUND when the field is completely missing.
   ```

---

### Problem: Inconsistent Formatting

**Symptoms:**
- Dates in different formats
- Some amounts with $, others without
- List separators vary (commas vs pipes)

**Solutions:**

1. **Add explicit format rules:**
   ```yaml
   extraction_rules: |
     CRITICAL FORMATTING RULES:
     - DATES: Always convert to DD/MM/YYYY format
     - MONEY: Always include $ and exactly 2 decimal places
     - LISTS: Always use " | " (space-pipe-space) separator
     - ABN: Always format as XX XXX XXX XXX with spaces
     - PHONE: Always include country code +XX
   ```

2. **Add examples in rules:**
   ```yaml
   extraction_rules: |
     FORMATTING EXAMPLES:
     - Date: "15/03/2025" not "March 15, 2025" or "2025-03-15"
     - Money: "$1,234.56" not "1234.56" or "$1234.5"
     - List: "Apples | Oranges | Bananas" not "Apples, Oranges, Bananas"
   ```

---

### Problem: Model Hallucinating Data

**Symptoms:**
- Extracted data not visible in document
- Invented dates, amounts, or names
- Unrealistic values

**Solutions:**

1. **Switch to strict or precise mode:**
   ```yaml
   default_system_mode: "strict"
   ```

2. **Add anti-hallucination rules:**
   ```yaml
   extraction_rules: |
     CRITICAL - NO HALLUCINATION:
     - Extract ONLY text that is physically present in the image
     - Do NOT invent, guess, or extrapolate any information
     - Do NOT fill in missing fields with assumed values
     - Do NOT use information from document type knowledge
     - If a field is not visible, always use NOT_FOUND
     - When uncertain, use NOT_FOUND
   ```

3. **Add verification requirement:**
   ```yaml
   extraction_rules: |
     VERIFICATION:
     - For every extracted value, ensure it is visible in the image
     - For dates: Must be explicitly shown, not inferred
     - For amounts: Must be printed, not calculated
     - For names: Must match exactly, not assumed from context
   ```

---

### Problem: Missing Line Items

**Symptoms:**
- Only first few items extracted
- Long lists truncated
- "..." appearing in item lists

**Solutions:**

1. **Add completeness requirement:**
   ```yaml
   document_instructions:
     invoice: |
       This is an INVOICE. CRITICAL REQUIREMENTS:
       - Extract EVERY SINGLE line item, no matter how many
       - Do NOT truncate or summarize line items
       - If there are 50 items, extract all 50
       - Count items to ensure completeness
   ```

2. **Increase max_new_tokens in model config:**
   ```python
   CONFIG = {
       'MAX_NEW_TOKENS': 4000,  # Increase from default 2048
       ...
   }
   ```

---

### Problem: Document Type Misclassification

**Symptoms:**
- Invoices detected as receipts
- Statements detected as invoices
- Wrong extraction template used

**Solutions:**

1. **Enhance detection prompt:**
   ```yaml
   detection_prompt: |
     Analyze this document image carefully.

     INVOICE indicators: Invoice number, due date, billing terms
     RECEIPT indicators: Transaction completed, payment received
     BANK STATEMENT indicators: Account balance, multiple transactions

     Respond with ONLY ONE of:
     - INVOICE (for bills awaiting payment)
     - RECEIPT (for completed transactions)
     - BANK_STATEMENT (for account activity summary)

     Format: DOCUMENT_TYPE: [type]
   ```

2. **Add distinguishing features:**
   ```yaml
   detection_prompt: |
     Classification guide:

     INVOICE = Bill requesting payment (has "Amount Due" or "Please Pay")
     RECEIPT = Proof of payment (has "Paid" or "Thank You" or "Transaction Complete")
     BANK_STATEMENT = Account history (has "Opening Balance", "Closing Balance")

     Look for these specific words and respond immediately.
   ```

---

## Best Practices

### 1. Start Conservative, Then Relax

```yaml
# Start with strict mode
default_system_mode: "strict"

# Review results, then progressively relax if needed
default_system_mode: "precise"  # Still conservative
default_system_mode: "expert"   # Balanced
default_system_mode: "flexible" # Liberal
```

**Why:** Easier to identify errors when being strict. You can always relax constraints based on results.

---

### 2. Be Specific in Instructions

**Bad (vague):**
```yaml
invoice: "Extract invoice data"
```

**Good (specific):**
```yaml
invoice: |
  This is an INVOICE. Extract these fields in order:
  1. Vendor name (top of document)
  2. Invoice number (often starts with "INV-")
  3. Invoice date (when invoice was issued)
  4. Due date (when payment is expected)
  5. Line items (EACH product/service sold)
  6. Total amount (final amount to pay)
```

---

### 3. Use Examples in Rules

**Bad:**
```yaml
extraction_rules: "Format dates correctly"
```

**Good:**
```yaml
extraction_rules: |
  Date formatting examples:
  - "15 Jan 2025" → "15/01/2025"
  - "2025-01-15" → "15/01/2025"
  - "January 15, 2025" → "15/01/2025"
  Always use DD/MM/YYYY format.
```

---

### 4. Test Incrementally

1. **Change one thing at a time**
2. **Test on known documents**
3. **Compare before/after results**
4. **Document what worked**

Example workflow:
```
Day 1: Test default settings, note issues
Day 2: Change system_mode to "precise", compare
Day 3: Add custom invoice instructions, compare
Day 4: Adjust extraction rules, compare
Day 5: Finalize best configuration
```

---

### 5. Keep a Configuration Log

Create a file `config/prompts_changelog.md`:

```markdown
# Prompt Configuration Changes

## 2025-11-18 - Initial Setup
- Using default expert mode
- Baseline accuracy: 74%

## 2025-11-19 - Added Construction Fields
- Updated invoice instructions for construction
- Added license number extraction
- Result: 82% accuracy on construction invoices

## 2025-11-20 - Fixed Date Formatting
- Added strict date formatting rule (DD/MM/YYYY)
- Result: Date accuracy improved from 67% to 95%
```

---

### 6. Use Comments Liberally

```yaml
# ==============================================================
# CONSTRUCTION INVOICE CONFIGURATION
# Last updated: 2025-11-18
# Optimized for: Australian construction industry
# Tested on: 1,000+ sample invoices
# Current accuracy: 85%
# ==============================================================

document_instructions:
  invoice: |
    # Focus on construction-specific fields
    # License numbers are critical for compliance
    This is a CONSTRUCTION INVOICE. Extract:
    - Contractor license (format: LIC-XXXXX)
    ...
```

---

### 7. Create Document-Type Variants

For different subtypes of the same document:

```yaml
document_instructions:
  invoice_construction: |
    Construction invoice instructions...

  invoice_medical: |
    Medical invoice instructions...

  invoice_retail: |
    Retail invoice instructions...
```

**Select in notebook:**
```python
# Temporarily override document type
CONFIG = {
    'DOCUMENT_TYPE_OVERRIDE': 'invoice_medical',
    ...
}
```

---

### 8. Validate Configuration Syntax

Before running, validate YAML syntax:

**Online validators:**
- https://www.yamllint.com/
- https://jsonformatter.org/yaml-validator

**Command line:**
```bash
python -c "import yaml; yaml.safe_load(open('config/prompts.yaml'))"
```

---

## Configuration Templates

### Template 1: High-Accuracy Financial Processing

```yaml
default_system_mode: "strict"

system_prompts:
  strict: "Extract ONLY information that is completely clear and unambiguous. Financial accuracy is critical."

extraction_rules: |
  STRICT FINANCIAL RULES:
  - For monetary values: Must include $ and exactly 2 decimal places
  - For amounts: Verify totals match line items (if discrepancy, mark with [ERROR])
  - For dates: Only DD/MM/YYYY format
  - For ABN: Must be exactly 11 digits formatted as XX XXX XXX XXX
  - Use NOT_FOUND for any field with even slight uncertainty
  - Never round or approximate values
  - Flag any calculations that don't reconcile

document_instructions:
  invoice: |
    FINANCIAL AUDIT INVOICE EXTRACTION:
    - Verify GST calculation (Total - GST = Subtotal)
    - Verify line item totals (Qty × Price = Line Total)
    - Extract payment terms explicitly
    - Note any early payment discounts
    - Flag if "PAID" stamp is present
```

---

### Template 2: High-Volume Retail Processing

```yaml
default_system_mode: "flexible"

system_prompts:
  flexible: "Extract efficiently from retail documents. Handle minor OCR errors gracefully."

extraction_rules: |
  RETAIL EFFICIENCY RULES:
  - Extract quickly, don't over-analyze
  - For store names: Accept common misspellings
  - For item descriptions: Abbreviate if too long
  - For prices: Accept even if OCR has minor errors
  - Use NOT_FOUND sparingly
  - Prioritize speed over perfection

document_instructions:
  receipt: |
    RETAIL RECEIPT FAST EXTRACTION:
    - Store name and location (approximate is fine)
    - Date and time
    - Top 10 purchased items (if more, summarize)
    - Total amount (critical - must be accurate)
    - Payment method
```

---

### Template 3: Legal Document Discovery

```yaml
default_system_mode: "strict"

system_prompts:
  strict: "Legal document extraction. Accuracy is paramount. Mark all uncertainties."

extraction_rules: |
  LEGAL DISCOVERY RULES:
  - Extract text verbatim (exact wording critical)
  - Preserve all punctuation and capitalization
  - For dates: Note if uncertain with [DATE UNCLEAR]
  - For signatures: Note presence even if illegible
  - For stamps/seals: Describe even if can't read text
  - For amendments/corrections: Note with [AMENDED] marker
  - For redactions: Mark as [REDACTED] not NOT_FOUND
  - Preserve document reference numbers exactly

document_instructions:
  invoice: |
    LEGAL INVOICE EXTRACTION:
    - Extract all parties' legal names (full entities)
    - Note any contract or matter references
    - Preserve all terms and conditions text
    - Extract authorized signatory information
    - Note any attachments referenced
    - Flag any dispute or collection notices
```

---

## Quick Reference Card

**Print this page for quick reference:**

```
┌─────────────────────────────────────────────────────────────┐
│              PROMPT CONFIGURATION QUICK GUIDE               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  FILE LOCATION: config/prompts.yaml                         │
│                                                             │
│  COMMON CHANGES:                                            │
│                                                             │
│  1. Change extraction mode:                                 │
│     default_system_mode: "expert"  # or precise, flexible   │
│                                                             │
│  2. Add document instructions:                              │
│     document_instructions:                                  │
│       invoice: |                                            │
│         [Your custom instructions here]                     │
│                                                             │
│  3. Modify extraction rules:                                │
│     extraction_rules: |                                     │
│       - [Your custom rules here]                            │
│                                                             │
│  SYSTEM MODES:                                              │
│    strict    → Highest accuracy, most NOT_FOUND             │
│    precise   → High accuracy, conservative                  │
│    expert    → Balanced (default)                           │
│    flexible  → More complete, handles poor quality          │
│    structured→ For standardized forms                       │
│                                                             │
│  COMMON RULES:                                              │
│    • Dates: "Use DD/MM/YYYY format"                         │
│    • Money: "Include $ and 2 decimals"                      │
│    • Lists: "Use | separator (pipe)"                        │
│    • ABN: "Format as XX XXX XXX XXX"                        │
│                                                             │
│  TROUBLESHOOTING:                                           │
│    Too many NOT_FOUND    → Use "flexible" mode              │
│    Inconsistent format   → Add format examples to rules     │
│    Missing data          → Check MAX_NEW_TOKENS in config   │
│    Wrong doc type        → Enhance detection_prompt         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Getting Help

### Questions to Ask

1. **"What accuracy am I getting with current settings?"**
   - Review the output CSV accuracy column
   - Compare against ground truth

2. **"Which fields are failing most often?"**
   - Check field-level accuracy breakdown
   - Focus prompt improvements on weak fields

3. **"Are extractions too strict or too loose?"**
   - Count NOT_FOUND occurrences
   - Review sample outputs manually

### Where to Get Support

- **Documentation:** `docs/` folder in project
- **Examples:** `evaluation_data/` sample documents
- **Logs:** Check notebook output for extraction details
- **Compare:** Look at `LMM_POC` working notebooks

---

## Conclusion

Prompt configuration gives you powerful control over document extraction **without programming**. Key takeaways:

✅ **Start with defaults**, then customize incrementally
✅ **Test changes** on known documents before production
✅ **Document your changes** in comments and change logs
✅ **Use specific examples** in instructions and rules
✅ **Balance accuracy vs. completeness** with system modes

Remember: The best configuration is one that works for **your specific documents and requirements**. Don't be afraid to experiment!

---

**Document Version:** 1.0
**Last Updated:** 2025-11-18
**Tested With:** InternVL3-8B, Llama-3.2-11B-Vision
**Configuration File:** `config/prompts.yaml`
