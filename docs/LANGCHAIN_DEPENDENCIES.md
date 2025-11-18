# LangChain Dependencies for Document Extraction

## Why LangChain?

LangChain provides a modular, type-safe framework for building production-ready vision-language model pipelines without coupling our business logic to specific model implementations. It enables us to wrap both Llama-3.2-Vision and InternVL3 models behind a unified interface, implement self-healing parsers that automatically retry failed extractions, leverage Pydantic schemas for type-safe field validation (eliminating ~400 lines of custom parsing code), and structure our extraction workflow into reusable chains (document detection → field extraction → validation) that can be easily tested, monitored, and swapped between different models without code changes. This architectural flexibility is critical for maintaining the document extraction system as models evolve and business requirements change.

## Required LangChain Packages

### Core Framework (Actually Used)

- **langchain-core** (≥1.0.0)
  - PyPI: https://pypi.org/project/langchain-core/
  - Purpose: Core abstractions and base classes (BaseChatModel, callbacks, prompts, output parsers)
  - Used in: All LangChain integration modules

- **langchain** (≥1.0.0)
  - PyPI: https://pypi.org/project/langchain/
  - Purpose: Main framework with self-healing output parsers (OutputFixingParser)
  - Used in: `common/langchain_parsers.py`

### Supporting Dependencies

- **pydantic** (≥2.0.0)
  - PyPI: https://pypi.org/project/pydantic/
  - Purpose: Type-safe data validation and schema definitions (required by LangChain)
  - Used in: `common/extraction_schemas.py`, all LangChain modules

## Installation

```bash
# Install via conda environment
conda env create -f environment.yml

# Or install manually with pip
pip install langchain>=1.0.0 \
            langchain-core>=1.0.0 \
            pydantic>=2.0.0
```

## Version Compatibility

- **Python**: 3.11
- **LangChain Core**: 1.0+
- **LangChain**: 1.0+
- **Pydantic**: 2.0+ (required for LangChain 1.0)

## Note

The following packages are listed in `environment.yml` but **not currently used** in the codebase:
- `langchain-community` (can be removed if not needed for future features)
- `langchain-text-splitters` (not applicable - we process images, not text documents)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-18
**Environment File:** `environment.yml`
