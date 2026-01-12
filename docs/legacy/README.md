# Legacy Gradience Components

⚠️ **DEPRECATED/EXPERIMENTAL**: These components are from the original Gradience prototype and are **not supported** in the current release.

## Current Release (vNext)

For the **current, stable Gradience release**, please refer to:
- [Main README](../../README.md) - Primary documentation
- [Quick Reference](../../QUICK_REFERENCE.md) - Copy-paste commands  
- [User Manual](../../USER_MANUAL.md) - Comprehensive guide

**Current API:** `gradience.vnext` with schema `gradience.vnext.telemetry/v1`

---

## Legacy Components

### 🚫 Deprecated Modules
These modules were experimental and are no longer maintained:
- `guard.py` - Training integrity monitoring (experimental)
- `controller.py` - High-level orchestration (replaced by vnext CLI)
- `huggingface.py` - Legacy HF integration (replaced by vnext.integrations.hf)

### 🚫 Deprecated Scripts  
Research scripts from the prototype phase:
- `run_*.py` - Various experimental training scripts with old telemetry
- These use legacy terminology (`kappa`, `Guard System`, etc.)
- **Do not use these** - they are incompatible with vNext

### 🚫 Deprecated Concepts
- **"Training Integrity Insurance"** → Now: Telemetry-first observability
- **"Guard System"** → Now: Conservative audit recommendations  
- **"κ̃ (kappa-tilde)"** → Now: Condition number monitoring
- **MIT License extras** → Now: Core functionality, no extras needed

---

## Migration Guide

### From Legacy to vNext

**OLD (deprecated):**
```python
# Don't use this
from gradience import Guard
guard = Guard(model, integrity=True)

# Or this
from gradience.integrations.huggingface import GradienceCallback
```

**NEW (current - canonical import):**
```python
# Use this canonical import path
from gradience.vnext.integrations.hf import GradienceCallback
trainer.add_callback(GradienceCallback())
```

### Command Migration

**OLD (deprecated):**
```bash
# Don't use
python run_finetune.py --guard-enabled
```

**NEW (current):**
```bash
# Use this workflow
gradience check --task seq_cls --peft-dir my_adapter  
gradience monitor run.jsonl
gradience audit --peft-dir my_adapter
```

---

## Why These Were Deprecated

1. **Experimental stability**: Guard system was single-GPU only and experimental
2. **Complex API**: Too many concepts (Guard, Controller, etc.) for the core use case
3. **Confusing positioning**: "Training integrity insurance" didn't clearly communicate value
4. **Schema instability**: Old telemetry format was not versioned properly

## vNext Improvements

1. **Simple, clear positioning**: "Telemetry-first observability for LoRA"
2. **Stable schema**: `gradience.vnext.telemetry/v1` is versioned and stable
3. **Conservative recommendations**: Focus on audit → suggest → validate workflow
4. **Framework integrations**: Drop-in callbacks for popular frameworks
5. **Minimal, testable**: Core functionality works on CPU, no heavy dependencies

---

## Archive Notice

These legacy components are preserved for reference only. They:
- ❌ Are not tested or maintained
- ❌ May have security or compatibility issues  
- ❌ Use deprecated terminology and concepts
- ❌ Are incompatible with current Gradience vNext

**For all current usage, please use the vNext API documented in the main README.**