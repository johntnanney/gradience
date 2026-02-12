# Release Checklist

**Version-critical items that must be updated for each release.**

## Pre-Release Version Updates

- [ ] Update version in `pyproject.toml`
- [ ] Update citation version in `README.md` (both BibTeX and APA)
- [ ] Update citation year if this is a new calendar year release
- [ ] Update BibTeX key year if needed (e.g., `gradience2026` → `gradience2027`)
- [ ] Add entry to `CHANGELOG.md` for new version
- [ ] Update any hardcoded version references in documentation

## Citation Block Consistency Check

Current citation format in README.md should match:

```bibtex
@software{gradienceYYYY,
  title = {Gradience: Evidence-Based LoRA Compression for Language Models},
  author = {Nanney, John T.},
  year = {YYYY},
  url = {https://github.com/gradience-ai/gradience},
  note = {Version X.Y.Z}
}
```

**APA Style:** Nanney, J. T. (YYYY). *Gradience: Evidence-based LoRA compression for language models* (Version X.Y.Z) [Computer software]. https://github.com/gradience-ai/gradience

Where:
- `YYYY` = Release year 
- `X.Y.Z` = Version from `pyproject.toml`
- BibTeX key = `gradience` + release year

## Release Process

1. [ ] Complete all pre-release version updates
2. [ ] Test citation formats render correctly on GitHub
3. [ ] Run pip install test with new version
4. [ ] Create git tag matching pyproject.toml version
5. [ ] Create GitHub release with changelog excerpt
6. [ ] Verify PyPI upload includes correct version
7. [ ] Test citation formats render correctly on PyPI

## Version Sources of Truth

- **Package version**: `pyproject.toml` (single source)
- **Citation year**: Calendar year of release
- **BibTeX key**: `gradience` + release year
- **Release notes**: `CHANGELOG.md` + GitHub releases

## Automation Opportunities

Consider automating these updates in CI/CD:
- Version bumping from pyproject.toml → README citation
- CHANGELOG.md date updates
- Git tag creation
- PyPI upload with version verification