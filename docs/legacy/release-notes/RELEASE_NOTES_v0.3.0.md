# Gradience v0.3.0 Release Notes

**Release Date**: January 13, 2025  
**Tag**: v0.3.0  
**Title**: Experimental LoRA Guard (snapshot + rollback)

## 🎯 Major Features

### 🛡️ Experimental LoRA Guard
- **Automatic rollback protection** for LoRA training
- Detects gradient explosions and NaN/Inf corruption
- Takes periodic snapshots of adapter weights
- Rolls back to stable states when triggers detected
- Built-in anti-thrash protection (cooldown & max rollbacks)
- **Zero interference when disabled** (validated by contract tests)

### 📊 Enhanced Telemetry
- **Comprehensive context fields** in all Guard events
- Standardized alert severity levels (INFO → WARNING → ERROR)
- Complete debugging information in JSONL
- Canonical event codes (GUARD_INIT, GUARD_ROLLBACK, etc.)

### 📚 Improved Documentation
- **Golden path integration** prominently displayed
- Copy-pasteable CLI commands
- Crystal clear Guard warnings
- Frozen public API surface (PUBLIC_API.md)
- CLI cheat sheet for quick reference

## 📦 What's New

### Files Added (19)
- `gradience/vnext/experimental/guard.py` - LoRA Guard implementation
- `tests/test_guard_non_interference_contract.py` - Release gate test
- `tests/helpers/hf_callback_driver.py` - Reusable test infrastructure
- `examples/vnext/golden_path.py` - One-line integration example
- `docs/experimental/guard.md` - Guard documentation
- `PUBLIC_API.md` - API stability guarantees
- `CLI_CHEATSHEET.md` - Command reference
- Multiple Guard validation examples and tests

### Files Modified (9)
- Enhanced HF integration (`gradience/vnext/integrations/hf.py`)
- Improved CLI with Guard visibility (`gradience/cli.py`)
- Updated README with golden path
- Version bump to 0.3.0

## ✅ Testing

- **82 unit tests** all passing
- **Smoke tests** validated (HF + standard)
- **Contract test** ensures Guard non-interference
- **CPU-friendly** validation suite

## 🚀 Quick Start

```python
from transformers import Trainer
from gradience.vnext.integrations.hf import GradienceCallback

trainer = Trainer(..., callbacks=[GradienceCallback()])
trainer.train()

# Telemetry: <output_dir>/run.jsonl
```

Then analyze:
```bash
gradience monitor <output_dir>/run.jsonl --verbose
gradience audit --peft-dir <output_dir>/adapter --layers
```

## ⚠️ Important Notes

### Guard is EXPERIMENTAL
- Not recommended for production use
- Can stop training and roll back weights
- Does NOT fix root causes (data bugs, bad objectives)
- Always validate improvements with eval metrics
- Disabled by default (`enable_guard=False`)

### Backward Compatibility
- All existing functionality preserved
- Public API surface now frozen
- Telemetry schema remains stable (v1)

## 📈 Statistics

- **884 lines added**, 119 removed (net +765)
- **19 new files**, 8 modified
- Comprehensive test coverage
- Zero breaking changes

## 🙏 Acknowledgments

This release establishes the foundation for experimental training protection while maintaining Gradience's core philosophy of restraint-first monitoring and conservative recommendations.

## 📝 Upgrade Instructions

```bash
pip install --upgrade gradience

# Or from source:
git fetch --tags
git checkout v0.3.0
pip install -e .
```

## 🔗 Links

- [Documentation](docs/)
- [Public API](PUBLIC_API.md)
- [CLI Cheat Sheet](CLI_CHEATSHEET.md)
- [Guard Documentation](docs/experimental/guard.md)
- [Examples](examples/vnext/)

---

**Note**: This is an experimental release. Guard functionality should be thoroughly tested in your environment before enabling in any important training runs.