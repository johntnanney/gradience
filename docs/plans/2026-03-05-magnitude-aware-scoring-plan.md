# Magnitude-Aware Scoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorder the merge verdict decision tree so Frobenius imbalance is detected before orthogonal-SAFE, fixing the blind spot where 8-20x magnitude mismatches go undetected.

**Architecture:** Single-file change to `verdicts.py` — add `imbalanced_frob` threshold field, insert new Branch 0 (Frobenius imbalance + low overlap), switch old Branch 4 to use `frobenius_ratio`. All downstream code (recommend, plan, execute) already handles IMBALANCED verdicts correctly.

**Tech Stack:** Python 3.10+, pytest, frozen dataclasses

---

### Task 1: Add `imbalanced_frob` threshold field

**Files:**
- Modify: `gradience/vnext/merge/verdicts.py:39-78` (VerdictThresholds class)
- Test: `tests/merge/test_verdicts.py`

**Step 1: Write the failing test**

Add to `TestVerdictThresholds` in `tests/merge/test_verdicts.py`:

```python
def test_imbalanced_frob_default(self):
    """New Frobenius imbalance threshold exists with sensible default."""
    t = VerdictThresholds()
    assert t.imbalanced_frob == 5.0

def test_imbalanced_frob_in_profiles(self):
    """Conservative/permissive profiles set imbalanced_frob."""
    c = VerdictThresholds.conservative()
    p = VerdictThresholds.permissive()
    assert c.imbalanced_frob < p.imbalanced_frob
    assert c.imbalanced_frob == 3.0
    assert p.imbalanced_frob == 10.0

def test_imbalanced_frob_in_to_dict(self):
    """imbalanced_frob appears in serialized dict."""
    d = VerdictThresholds().to_dict()
    assert d["imbalanced_frob"] == 5.0
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/merge/test_verdicts.py::TestVerdictThresholds::test_imbalanced_frob_default -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'imbalanced_frob'` or `AttributeError`

**Step 3: Add `imbalanced_frob` field to VerdictThresholds**

In `gradience/vnext/merge/verdicts.py`, add the new field to the dataclass and update the profiles:

```python
@dataclass(frozen=True)
class VerdictThresholds:
    """Tunable thresholds for verdict decisions.

    Attributes
    ----------
    low_overlap : below this -> orthogonal subspaces
    high_overlap : above this -> significant shared subspace
    aligned : agreement above this -> same direction (redundant)
    conflicting : agreement below this -> opposing (conflicting)
    imbalanced : magnitude ratio above this -> imbalanced (legacy, sigma-1)
    imbalanced_frob : Frobenius norm ratio above this -> imbalanced
    """

    low_overlap: float = 0.2
    high_overlap: float = 0.5
    aligned: float = 0.5
    conflicting: float = -0.3
    imbalanced: float = 5.0
    imbalanced_frob: float = 5.0

    @classmethod
    def conservative(cls) -> VerdictThresholds:
        """Flag more potential issues. Good for early adoption."""
        return cls(
            low_overlap=0.15,
            high_overlap=0.35,
            aligned=0.6,
            conflicting=-0.2,
            imbalanced=3.0,
            imbalanced_frob=3.0,
        )

    @classmethod
    def permissive(cls) -> VerdictThresholds:
        """Flag only obvious problems. For experienced users."""
        return cls(
            low_overlap=0.3,
            high_overlap=0.7,
            aligned=0.4,
            conflicting=-0.5,
            imbalanced=10.0,
            imbalanced_frob=10.0,
        )
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/merge/test_verdicts.py::TestVerdictThresholds -v`
Expected: All TestVerdictThresholds tests PASS (including 3 new ones)

**Step 5: Commit**

```bash
git add gradience/vnext/merge/verdicts.py tests/merge/test_verdicts.py
git commit -m "feat(verdicts): add imbalanced_frob threshold field"
```

---

### Task 2: Write failing tests for new branch ordering

**Files:**
- Test: `tests/merge/test_verdicts.py`

**Step 1: Write the failing tests**

Add a new test class `TestFrobeniusImbalance` to `tests/merge/test_verdicts.py`:

```python
class TestFrobeniusImbalance:
    """Tests for Frobenius-based imbalance detection (Branch 0)."""

    def test_orthogonal_and_imbalanced(self):
        """Low overlap + high Frobenius ratio -> IMBALANCED (was SAFE before fix)."""
        metrics = _make_metrics(
            mean_overlap=0.05,
            max_overlap=0.1,
            frobenius_ratio=10.0,
            frobenius_norm_a=100.0,
            frobenius_norm_b=10.0,
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.IMBALANCED
        assert lv.suggested_strategy == "linear"
        assert lv.suggested_coefficients is not None
        # Coefficients should sum to 1.0
        assert sum(lv.suggested_coefficients) == pytest.approx(1.0, abs=0.01)
        # Weaker adapter (B, norm=10) should get higher coefficient
        coeff_a, coeff_b = lv.suggested_coefficients
        assert coeff_b > coeff_a

    def test_high_overlap_aligned_imbalanced_gets_redundant(self):
        """High overlap + aligned + Frobenius imbalanced -> REDUNDANT (overlap wins)."""
        metrics = _make_metrics(
            mean_overlap=0.8,
            max_overlap=0.9,
            directional_agreement=0.8,
            principal_angle_cosines=(0.9, 0.8, 0.7),
            frobenius_ratio=10.0,
            frobenius_norm_a=100.0,
            frobenius_norm_b=10.0,
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.REDUNDANT

    def test_high_overlap_opposing_imbalanced_gets_conflicting(self):
        """High overlap + opposing + Frobenius imbalanced -> CONFLICTING (overlap wins)."""
        metrics = _make_metrics(
            mean_overlap=0.8,
            max_overlap=0.9,
            directional_agreement=-0.7,
            principal_angle_cosines=(0.9, 0.8, 0.7),
            frobenius_ratio=10.0,
            frobenius_norm_a=100.0,
            frobenius_norm_b=10.0,
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.CONFLICTING

    def test_below_frob_threshold_stays_safe(self):
        """Low overlap + below Frobenius threshold -> SAFE (not imbalanced)."""
        metrics = _make_metrics(
            mean_overlap=0.05,
            max_overlap=0.1,
            frobenius_ratio=3.0,
            frobenius_norm_a=15.0,
            frobenius_norm_b=5.0,
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.SAFE

    def test_moderate_overlap_imbalanced(self):
        """Moderate overlap (between low and high) + Frobenius imbalanced -> IMBALANCED."""
        metrics = _make_metrics(
            mean_overlap=0.3,
            max_overlap=0.4,
            directional_agreement=0.2,
            frobenius_ratio=8.0,
            frobenius_norm_a=80.0,
            frobenius_norm_b=10.0,
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.IMBALANCED

    def test_coefficients_use_frobenius_ratio(self):
        """Rebalanced coefficients are derived from frobenius_ratio."""
        metrics = _make_metrics(
            mean_overlap=0.05,
            frobenius_ratio=20.0,
            frobenius_norm_a=200.0,
            frobenius_norm_b=10.0,
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.IMBALANCED
        coeff_a, coeff_b = lv.suggested_coefficients
        # ratio=20: strong gets 1/(1+20)=0.0476, weak gets 20/(1+20)=0.952
        assert coeff_a == pytest.approx(1.0 / 21.0, abs=0.01)
        assert coeff_b == pytest.approx(20.0 / 21.0, abs=0.01)

    def test_conservative_catches_more_imbalance(self):
        """Conservative profile (imbalanced_frob=3.0) catches smaller ratios."""
        metrics = _make_metrics(
            mean_overlap=0.05,
            frobenius_ratio=4.0,
            frobenius_norm_a=40.0,
            frobenius_norm_b=10.0,
        )
        # Default (threshold=5.0) -> SAFE
        lv_default = assess_layer("test.q_proj", "attn", metrics)
        assert lv_default.verdict == CompatibilityVerdict.SAFE

        # Conservative (threshold=3.0) -> IMBALANCED
        lv_conservative = assess_layer(
            "test.q_proj", "attn", metrics,
            thresholds=VerdictThresholds.conservative(),
        )
        assert lv_conservative.verdict == CompatibilityVerdict.IMBALANCED

    def test_high_overlap_remainder_imbalanced(self):
        """High overlap + not aligned/opposing + Frobenius imbalanced -> IMBALANCED (Branch 4)."""
        metrics = _make_metrics(
            mean_overlap=0.6,
            max_overlap=0.7,
            directional_agreement=0.2,  # between aligned and conflicting
            principal_angle_cosines=(0.7, 0.6, 0.5),
            frobenius_ratio=10.0,
            frobenius_norm_a=100.0,
            frobenius_norm_b=10.0,
        )
        lv = assess_layer("test.q_proj", "attn", metrics)

        assert lv.verdict == CompatibilityVerdict.IMBALANCED
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/merge/test_verdicts.py::TestFrobeniusImbalance -v`
Expected: FAIL — `test_orthogonal_and_imbalanced` gets SAFE instead of IMBALANCED, etc.

**Step 3: Commit failing tests**

```bash
git add tests/merge/test_verdicts.py
git commit -m "test(verdicts): add failing tests for Frobenius imbalance branch"
```

---

### Task 3: Implement new branch ordering in `assess_layer()`

**Files:**
- Modify: `gradience/vnext/merge/verdicts.py:136-274` (assess_layer function)

**Step 1: Implement the branch reordering**

Replace the `assess_layer` function body in `gradience/vnext/merge/verdicts.py`:

```python
def assess_layer(
    layer_name: str,
    module_type: str,
    metrics: SubspaceMetrics,
    thresholds: Optional[VerdictThresholds] = None,
) -> LayerVerdict:
    """Six-branch decision tree for layer-level verdict.

    Branch ordering prioritises magnitude imbalance (Frobenius-based) before
    orthogonal-safe, so that adapters with large energy mismatches get
    rebalanced coefficients even when their subspaces are nearly orthogonal.

    When overlap is high, the overlap-based branches (REDUNDANT, CONFLICTING)
    take priority because subspace interaction is the dominant concern.
    """
    if thresholds is None:
        thresholds = VerdictThresholds()

    # --- Branch 0 (NEW): Frobenius imbalanced + low-to-moderate overlap ---
    if (
        metrics.frobenius_ratio > thresholds.imbalanced_frob
        and metrics.mean_overlap < thresholds.high_overlap
    ):
        ratio = metrics.frobenius_ratio
        coeff_strong = 1.0 / (1.0 + ratio)
        coeff_weak = ratio / (1.0 + ratio)

        # Map strong/weak to (coeff_a, coeff_b) based on which is larger
        if metrics.frobenius_norm_a >= metrics.frobenius_norm_b:
            coefficients = (coeff_strong, coeff_weak)
        else:
            coefficients = (coeff_weak, coeff_strong)

        return LayerVerdict(
            layer_name=layer_name,
            module_type=module_type,
            metrics=metrics,
            verdict=CompatibilityVerdict.IMBALANCED,
            confidence=_overlap_confidence(
                metrics.frobenius_ratio, thresholds.imbalanced_frob
            ),
            recommendation=(
                f"Frobenius imbalance ({ratio:.1f}x, "
                f"norms: {metrics.frobenius_norm_a:.1f} vs "
                f"{metrics.frobenius_norm_b:.1f}). "
                f"The weaker adapter's contribution will be drowned out "
                f"with equal coefficients. Suggested rebalancing: "
                f"A={coefficients[0]:.3f}, B={coefficients[1]:.3f}."
            ),
            conflict_dimensions=0,
            safe_merge_rank=max(metrics.effective_rank_a, metrics.effective_rank_b),
            suggested_strategy="linear",
            suggested_coefficients=coefficients,
        )

    # --- Branch 1: Orthogonal subspaces ---
    if metrics.mean_overlap < thresholds.low_overlap:
        return LayerVerdict(
            layer_name=layer_name,
            module_type=module_type,
            metrics=metrics,
            verdict=CompatibilityVerdict.SAFE,
            confidence=_overlap_confidence(
                metrics.mean_overlap, thresholds.low_overlap
            ),
            recommendation=(
                f"Orthogonal subspaces (overlap={metrics.mean_overlap:.3f}). "
                f"Safe to merge with any method. Combined effective rank: "
                f"{metrics.effective_rank_a + metrics.effective_rank_b}."
            ),
            conflict_dimensions=0,
            safe_merge_rank=metrics.effective_rank_a + metrics.effective_rank_b,
            suggested_strategy="linear",
            suggested_coefficients=(0.5, 0.5),
        )

    # --- Branch 2: Redundant (high overlap, aligned) ---
    if (
        metrics.mean_overlap > thresholds.high_overlap
        and metrics.directional_agreement > thresholds.aligned
    ):
        return LayerVerdict(
            layer_name=layer_name,
            module_type=module_type,
            metrics=metrics,
            verdict=CompatibilityVerdict.REDUNDANT,
            confidence=min(
                _overlap_confidence(metrics.mean_overlap, thresholds.high_overlap),
                abs(metrics.directional_agreement - thresholds.aligned),
            ),
            recommendation=(
                f"High redundancy (overlap={metrics.mean_overlap:.3f}, "
                f"agreement={metrics.directional_agreement:.3f}). "
                f"Adapters learn similar features. TIES recommended to "
                f"deduplicate shared directions. Merged rank ~ "
                f"{max(metrics.effective_rank_a, metrics.effective_rank_b)}."
            ),
            conflict_dimensions=0,
            safe_merge_rank=max(metrics.effective_rank_a, metrics.effective_rank_b),
            suggested_strategy="ties",
            suggested_coefficients=(0.5, 0.5),
        )

    # --- Branch 3: Conflicting (high overlap, opposing) ---
    if (
        metrics.mean_overlap > thresholds.high_overlap
        and metrics.directional_agreement < thresholds.conflicting
    ):
        n_conflict = sum(
            1
            for cos_a in metrics.principal_angle_cosines
            if cos_a > thresholds.high_overlap
        )

        return LayerVerdict(
            layer_name=layer_name,
            module_type=module_type,
            metrics=metrics,
            verdict=CompatibilityVerdict.CONFLICTING,
            confidence=min(
                _overlap_confidence(metrics.mean_overlap, thresholds.high_overlap),
                abs(metrics.directional_agreement - thresholds.conflicting),
            ),
            recommendation=(
                f"CONFLICT: {n_conflict} shared direction(s) have opposing effects "
                f"(overlap={metrics.mean_overlap:.3f}, "
                f"agreement={metrics.directional_agreement:.3f}). "
                f"Direct merging will cause cancellation. Options: "
                f"(1) Use DARE with high drop rate, "
                f"(2) Reduce merge coefficient for weaker adapter, "
                f"(3) Exclude this layer from merge."
            ),
            conflict_dimensions=n_conflict,
            safe_merge_rank=metrics.effective_rank_a,
            suggested_strategy="dare",
            suggested_coefficients=None,
        )

    # --- Branch 4: Frobenius imbalanced (high-overlap remainder) ---
    if metrics.frobenius_ratio > thresholds.imbalanced_frob:
        ratio = metrics.frobenius_ratio
        coeff_strong = 1.0 / (1.0 + ratio)
        coeff_weak = ratio / (1.0 + ratio)

        if metrics.frobenius_norm_a >= metrics.frobenius_norm_b:
            coefficients = (coeff_strong, coeff_weak)
        else:
            coefficients = (coeff_weak, coeff_strong)

        return LayerVerdict(
            layer_name=layer_name,
            module_type=module_type,
            metrics=metrics,
            verdict=CompatibilityVerdict.IMBALANCED,
            confidence=_overlap_confidence(
                metrics.frobenius_ratio, thresholds.imbalanced_frob
            ),
            recommendation=(
                f"Frobenius imbalance ({ratio:.1f}x, "
                f"norms: {metrics.frobenius_norm_a:.1f} vs "
                f"{metrics.frobenius_norm_b:.1f}). "
                f"The weaker adapter's contribution will be drowned out "
                f"with equal coefficients. Suggested rebalancing: "
                f"A={coefficients[0]:.3f}, B={coefficients[1]:.3f}."
            ),
            conflict_dimensions=0,
            safe_merge_rank=max(metrics.effective_rank_a, metrics.effective_rank_b),
            suggested_strategy="linear",
            suggested_coefficients=coefficients,
        )

    # --- Branch 5: Moderate / ambiguous -> default SAFE ---
    return LayerVerdict(
        layer_name=layer_name,
        module_type=module_type,
        metrics=metrics,
        verdict=CompatibilityVerdict.SAFE,
        confidence=0.5,
        recommendation=(
            f"Moderate subspace interaction (overlap={metrics.mean_overlap:.3f}, "
            f"agreement={metrics.directional_agreement:.3f}). "
            f"Standard merge methods should work. TIES is a safe default."
        ),
        conflict_dimensions=0,
        safe_merge_rank=(metrics.effective_rank_a + metrics.effective_rank_b) // 2,
        suggested_strategy="ties",
        suggested_coefficients=(0.5, 0.5),
    )
```

**Step 2: Update the module docstring**

Replace the docstring at top of `gradience/vnext/merge/verdicts.py` (lines 1-16):

```python
"""
Decision logic for merge compatibility.

Translates raw SubspaceMetrics into actionable per-layer assessments and
an aggregate compatibility verdict.  The decision tree follows a six-branch
structure with Frobenius imbalance checked first:

    0. Frobenius imbalance + low/moderate overlap -> IMBALANCED (rebalanced)
    1. Low overlap  -> SAFE (orthogonal subspaces)
    2. High overlap + aligned  -> REDUNDANT (de-dup needed)
    3. High overlap + opposing -> CONFLICTING (danger zone)
    4. Frobenius imbalance + high overlap (remainder) -> IMBALANCED
    5. Everything else         -> SAFE with moderate confidence

Thresholds ship as defaults but can be overridden via CLI flags or a
VerdictThresholds instance.
"""
```

**Step 3: Run all tests**

Run: `python3 -m pytest tests/merge/test_verdicts.py -v`
Expected: ALL PASS (16 existing + 3 new threshold tests + 8 new branch tests = 27 total)

**Step 4: Commit**

```bash
git add gradience/vnext/merge/verdicts.py tests/merge/test_verdicts.py
git commit -m "feat(verdicts): reorder branches for Frobenius-first imbalance detection

Move Frobenius imbalance check to Branch 0 (before orthogonal-SAFE).
When overlap is below high_overlap, magnitude imbalance now fires
immediately with rebalanced coefficients. High-overlap layers still
fall through to REDUNDANT/CONFLICTING branches.

Fixes the Study 16 blind spot where 8-20x Frobenius norm ratios
produced SAFE verdicts with 50/50 coefficients."
```

---

### Task 4: Fix existing imbalanced test

**Files:**
- Modify: `tests/merge/test_verdicts.py`

The existing `test_imbalanced` test (line 132) uses `magnitude_ratio=8.0` with
`mean_overlap=0.3` (moderate). With the new Branch 0, this test now also needs
`frobenius_ratio` set to trigger imbalance. The test currently passes because
`_make_metrics` defaults `frobenius_ratio=1.1`, so Branch 0 won't fire. But the
old Branch 4 used `magnitude_ratio`, which is no longer checked. Update the test
to use `frobenius_ratio` instead:

**Step 1: Update the existing test**

```python
def test_imbalanced(self):
    """Extreme Frobenius ratio -> IMBALANCED."""
    metrics = _make_metrics(
        mean_overlap=0.3,  # moderate overlap (between low and high)
        max_overlap=0.4,
        directional_agreement=0.2,
        frobenius_ratio=8.0,
        frobenius_norm_a=80.0,
        frobenius_norm_b=10.0,
    )
    lv = assess_layer("test.q_proj", "attn", metrics)

    assert lv.verdict == CompatibilityVerdict.IMBALANCED
    assert lv.suggested_coefficients is not None
    assert sum(lv.suggested_coefficients) == pytest.approx(1.0, abs=0.01)
```

**Step 2: Run all tests**

Run: `python3 -m pytest tests/merge/test_verdicts.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/merge/test_verdicts.py
git commit -m "test(verdicts): update imbalanced test to use frobenius_ratio"
```

---

### Task 5: Run full test suite and verify no regressions

**Files:** None (verification only)

**Step 1: Run the full merge test suite**

Run: `python3 -m pytest tests/merge/ -v`
Expected: ALL PASS

**Step 2: Run the full project test suite**

Run: `python3 -m pytest tests/ -v --timeout=60`
Expected: ALL PASS (or only pre-existing failures unrelated to verdicts)

**Step 3: Commit (squash if needed)**

If any test adjustments were needed, commit them:

```bash
git add -A
git commit -m "fix: adjust tests for Frobenius imbalance branch ordering"
```
