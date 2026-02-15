# Gradience Release Checklist

Internal checklist to ensure consistent releases and protect empirical validation work.

## Pre-Release Validation

### Core Functionality
- [ ] **Core tests pass**: `python -m pytest tests/ -v`
- [ ] **Smoke tests pass**: `./scripts/smoke.sh` (if exists)
- [ ] **Integration tests**: Verify HuggingFace callback integration
- [ ] **Documentation up to date**: README, API docs, examples

### Policy Regression Protection
- [ ] **Policy regression tests pass**: `./scripts/test_policy_regression.sh`
- [ ] **Policy files consistent**: Machine-readable YAML matches prose documentation
- [ ] **No accidental policy drift**: r=20 still primary, r=16 still marked unsafe

### Bench Validation (If Applicable)

#### Required for Rank Suggestion Changes
If the release modifies rank suggestion algorithms, compression protocols, or safety thresholds:

- [ ] **Re-run mini-validation**: `python -m gradience.bench.run_bench --config gradience/bench/configs/distilbert_sst2_mini_validation.yaml`
- [ ] **Verify safety policy compliance**: Updated algorithms don't violate existing safety criteria
- [ ] **Update empirical evidence**: If validation results change, update policy YAML with new evidence
- [ ] **Update documentation**: VALIDATION_POLICY.md reflects any methodology changes

#### Required for Policy Changes
If updating safety policies or validation criteria:

- [ ] **Update VALIDATION_POLICY.md**: Document policy changes and rationale
- [ ] **Update policy YAML**: `gradience/bench/policies/safe_uniform.yaml` with new thresholds/evidence  
- [ ] **Update regression tests**: Adjust expected values in `tests/test_bench/test_policy_regression.py`
- [ ] **Document breaking changes**: Note policy changes in release notes if they affect users

### Version and Release Notes
- [ ] **Version updated**: `pyproject.toml`, `__init__.py` version strings
- [ ] **Release notes drafted**: Document new features, breaking changes, policy updates
- [ ] **Breaking changes highlighted**: Especially any safety policy modifications

## Release Process

### Tagging and Publishing
- [ ] **Create git tag**: `git tag v0.x.y`
- [ ] **Build package**: `python -m build`
- [ ] **Test package**: Install in clean environment and verify
- [ ] **Publish to PyPI**: `twine upload dist/*`
- [ ] **Push tag**: `git push origin v0.x.y`

### Post-Release
- [ ] **GitHub release**: Create release with notes
- [ ] **Update documentation**: If hosted separately
- [ ] **Announce**: Internal/external communications if applicable

## Policy Change Guidelines

### When Policy Updates Required
Policy updates are needed when:
- New empirical validation data contradicts current recommendations
- Safety thresholds prove insufficient in production
- Extended validation (Certifiable level) upgrades confidence levels
- New tasks/models require policy expansion

### Policy Update Process
1. **Run comprehensive validation**: New benchmarks with proper statistical rigor
2. **Document empirical evidence**: Update policy YAML with complete validation results  
3. **Update regression tests**: Modify expected values to match new policy
4. **Update prose documentation**: Ensure VALIDATION_POLICY.md and README consistency
5. **Version policy**: Increment policy version in YAML metadata

### Emergency Policy Changes
For urgent safety issues:
1. **Immediate hotfix**: Update policy files and regression tests
2. **Emergency release**: Skip normal validation if safety-critical
3. **Post-hoc validation**: Run comprehensive validation after hotfix
4. **Documentation update**: Explain emergency change rationale

---

**Note**: This checklist protects empirical calibration work from accidental regression. The bench validation steps ensure that algorithmuc changes don't silently undo carefully calibrated safety decisions.