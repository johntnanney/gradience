# CLI Reference Summary

## Overview

Created a comprehensive CLI reference that matches the shipped package and serves as the definitive command-line documentation for Gradience.

## What Was Created

### 📖 **[docs/cli.md](cli.md)** - Complete CLI Reference

**Comprehensive documentation covering both core and benchmarking CLIs with real examples.**

## Content Coverage

### ✅ **Main CLI: gradience**

**Complete subcommand documentation:**
- **Overview**: All 7 subcommands (verify, report, check, audit, explain, truncate, monitor)
- **gradience audit**: Most detailed - all 20+ options with examples
- **gradience monitor**: Telemetry analysis options
- **gradience check**: Pre-flight validation options

**Key features documented:**
- Required vs optional arguments
- Output formats (JSON vs pretty text)
- UDR computation options
- Rank policy configuration
- Integration with telemetry (--append)

### ✅ **Benchmarking CLI: gradience-bench**

**Captured from actual CLI help output:**
- All command-line options with exact descriptions
- Required vs optional argument breakdown
- Mode control (smoke, CI, full-mode)
- Device selection (cpu, mps, cuda)
- Candidate control (max-candidates)

### ✅ **Fast Mode vs Full Mode (Critical Documentation)**

**Clear explanation of policy differences:**

**Fast Mode (Default):**
- Policies: `energy_p90`, `knee_p90`, `erank_p90` only
- Candidates: ~3 compression variants
- Use case: Standard validation, CI pipelines

**Full Mode:**
- Policies: All available policy variants + legacy suggestions
- Candidates: All policies (capped at max-candidates)
- Use case: Research, comprehensive comparison

### ✅ **Resume Functionality** 

**Complete documentation of state management:**
- How resume works (stage tracking in `stage_state.json`)
- What gets skipped (probe training, downloads)
- Safety mechanisms (config validation, artifact verification)
- Exact JSON structure of state file
- Resume examples and troubleshooting

### ✅ **Exit Codes for Automation**

**Critical for CI/automation integration:**

| Exit Code | Meaning | When |
|-----------|---------|------|
| 0 | Success | All operations completed |
| 1 | General failure | Config errors, benchmark failures |
| 2 | Undertrained probe | Quality threshold not met |
| 130 | User interruption | Ctrl+C received |

**CI Mode specific exit codes:**
- 0: At least one strategy passed
- 1: No strategies passed validation

### ✅ **Integration Examples**

**Real-world automation patterns:**
- **Shell scripts**: Error handling, exit code checking
- **Python integration**: subprocess patterns, JSON parsing
- **GitHub Actions**: Complete CI workflow
- **Production deployment**: Resume-enabled benchmarking

### ✅ **Performance Guidance**

**Development vs production patterns:**
- Fast development: `--smoke --device cpu --resume`
- CI pipelines: `--ci --device cpu`
- Production validation: `--full-mode --resume --ci`

## Key Features

### **1. Matches Shipped Package**
- All help output captured from actual CLI
- Option descriptions exactly as implemented
- Examples tested and verified

### **2. Automation-Ready**
- Complete exit code documentation
- JSON output patterns
- Integration examples for common CI systems

### **3. Fast Mode Documentation**
- Explains what policies are tested in each mode
- Clear use case guidance
- Performance implications

### **4. Resume System**
- State storage location and format
- Safety mechanisms explained
- Troubleshooting for common issues

### **5. Real Examples**
- Copy-paste commands for common scenarios
- Shell script patterns
- Python integration code
- CI/CD workflow templates

## Impact

### **Before (Missing CLI Documentation):**
- Users had to run `--help` to understand options
- No explanation of fast vs full mode differences
- Exit codes undocumented (bad for automation)
- Resume functionality unclear
- No integration examples

### **After (Professional CLI Documentation):**
- Complete reference matching shipped package
- Clear mode differences and use cases
- Automation-ready exit code documentation
- Resume system fully explained
- Production-ready integration examples

### **User Experience:**
- **New users**: Can understand all options without trial-and-error
- **CI engineers**: Have complete exit code and automation guidance
- **Researchers**: Understand fast vs full mode trade-offs
- **Production teams**: Can implement resume-enabled workflows

## Validation

- ✅ All CLI help output captured from actual shipped package
- ✅ Exit codes verified from source code
- ✅ Fast vs full mode behavior documented from implementation
- ✅ Resume state format matches actual JSON structure
- ✅ Integration examples tested and validated
- ✅ Performance recommendations match actual behavior

## README Integration

- **Added CLI Reference link** in documentation section
- **Positioned prominently** after Installation Guide
- **Maintains logical flow**: Install → CLI → Cheatsheet → Guides

The CLI reference now serves as the definitive command-line documentation that matches the professional quality of the package itself!