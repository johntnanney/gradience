# Sidecar Makefile — one-command regeneration and validation
#
# Usage:
#   make all          # Regenerate everything
#   make analysis     # Just rerun analysis scripts
#   make figures      # Just regenerate figures
#   make validate     # Check that all expected outputs exist

PYTHON ?= python
BENCHMARKS = benchmarks
RESULTS = results/s01
FIGURES = figures

.PHONY: all analysis figures validate clean

all: analysis figures validate

analysis:
	@echo "── Running adjudication compilation ──"
	$(PYTHON) $(BENCHMARKS)/compile_adjudication.py
	@echo ""
	@echo "── Running instability analysis ──"
	$(PYTHON) $(BENCHMARKS)/instability_analysis.py
	@echo ""
	@echo "── Generating instability case table ──"
	$(PYTHON) $(BENCHMARKS)/gen_instability_case_table.py

figures:
	@echo "── Generating backbone shift figure ──"
	$(PYTHON) $(BENCHMARKS)/gen_backbone_shift_figure.py
	@echo ""
	@echo "── Generating seed stability figure ──"
	$(PYTHON) $(BENCHMARKS)/gen_seed_stability_figure.py
	@echo ""
	@echo "── Generating instability figures ──"
	$(PYTHON) $(BENCHMARKS)/gen_instability_figures.py
	@echo ""
	@echo "── Generating summary panel ──"
	$(PYTHON) $(BENCHMARKS)/gen_summary_panel.py

validate:
	@echo "── Validating outputs ──"
	@test -f $(RESULTS)/three_backbone_comparison.json || (echo "MISSING: three_backbone_comparison.json" && exit 1)
	@test -f $(RESULTS)/seed_stability.json || (echo "MISSING: seed_stability.json" && exit 1)
	@test -f $(RESULTS)/instability_profiles.json || (echo "MISSING: instability_profiles.json" && exit 1)
	@test -f $(RESULTS)/taxonomy.json || (echo "MISSING: taxonomy.json" && exit 1)
	@test -f $(RESULTS)/regime_summaries.json || (echo "MISSING: regime_summaries.json" && exit 1)
	@test -f $(RESULTS)/instability_case_table.json || (echo "MISSING: instability_case_table.json" && exit 1)
	@test -f $(RESULTS)/instability_case_table.md || (echo "MISSING: instability_case_table.md" && exit 1)
	@test -f $(FIGURES)/s01_backbone_shift.svg || (echo "MISSING: s01_backbone_shift.svg" && exit 1)
	@test -f $(FIGURES)/s01_seed_stability.svg || (echo "MISSING: s01_seed_stability.svg" && exit 1)
	@test -f $(FIGURES)/s01_taxonomy_scatter.svg || (echo "MISSING: s01_taxonomy_scatter.svg" && exit 1)
	@test -f $(FIGURES)/s01_regime_contrast.svg || (echo "MISSING: s01_regime_contrast.svg" && exit 1)
	@test -f $(FIGURES)/s01_summary_panel.svg || (echo "MISSING: s01_summary_panel.svg" && exit 1)
	@echo "All outputs present. ✓"

clean:
	rm -f $(RESULTS)/*.json
	rm -f $(FIGURES)/s01_*.svg $(FIGURES)/s01_*.png
