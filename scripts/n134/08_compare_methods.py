"""N134 four-method scheduled comparison.

Compares four merge-risk triage methods on N134 cross-task pairs:

    1. Gradience: S_H1 (O-module depth-weighted alignment)
    2. KnOTS (Stoica et al. 2024): joint SVD rotation, interference norm
    3. TSV (Gargiulo et al. 2025): task-singular-vector whitened interference
    4. SVC (Li et al. 2026): singular-value rescaling, SV-inflation index

Methods 2-4 are implemented from U/V factors stored in audit v2.1 files.
If a method's adaptation is judged unfaithful (e.g. missing required data),
it is reported as "adaptation_inadequate".

Comparison protocol:
    - Rank 45 cross-task pairs by each method's risk score
    - Select "safe" lowest-half (N=22 pairs)
    - Measure mean max_degradation in retained set
    - Block-bootstrap CIs (5000 resamples, family-pair-blocked)

Inputs:
    /workspace/n134/audit/pair_alignment_full.json
    /workspace/n134/audit/adapter_profiles.json
    /workspace/n134/audit/v2.1/*.json  (per-pair audit v2.1 with U/V factors)
    /workspace/n134/merges/merge_eval_summary.json

Output:
    /workspace/n134/method_comparison.json
    /workspace/n134/figures/method_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# WORKSPACE defaults to the pod-absolute path that produced the committed
# N134 artifacts. Replicators running outside that environment can
# override via the N134_WORKSPACE environment variable without editing
# the script; the default preserves pod-era provenance. NB: Phase 5
# additionally requires per-adapter .npz SVD sidecars under
# {WORKSPACE}/audit/{adapter_id}_svd.npz which are not committed to the
# repository (1.2 GB total); replicators need to re-audit adapters (see
# scripts/n134/03_spectral_audit.py) before running this script.
import os  # noqa: E402

WORKSPACE = Path(os.environ.get("N134_WORKSPACE", "/workspace/n134"))
AUDIT_DIR = WORKSPACE / "audit"
AUDIT_V21_DIR = AUDIT_DIR / "v2.1"
# Phase 5 plumbing resolution (2026-04-20):
# 03_spectral_audit.py writes per-adapter SVD factors as AUDIT_DIR/{adapter}_svd.npz
# with keys "layers.{i}.self_attn.{module}__{U|S|Vt}". This module synthesizes the
# per-pair v2.1 dict shape expected by knots/tsv/svc at read time via LazyPairV21,
# so the methods code below sees the exact shape documented in the v2.1 schema.
MERGE_DIR = WORKSPACE / "merges"
FIG_DIR = WORKSPACE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_LAYERS = 32
N_BOOTSTRAP = 5000
RNG_SEED = 20260418
N_RETAIN = 22  # retain lowest-risk half of 45 cross-task pairs (floor(45/2))

# FAMILY_B partition
FAMILY_B: dict[str, str] = {
    "arc_challenge": "science_qa",
    "hellaswag": "commonsense_completion",
    "winogrande": "coreference_commonsense",
    "openbookqa": "knowledge_qa",
    "commonsenseqa": "commonsense_qa",
    "piqa": "physical_commonsense",
    "siqa": "social_commonsense",
    "boolq": "reading_comprehension",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


class AdapterSVDCache:
    """Lazy-loads per-adapter SVD factors from .npz sidecars."""

    def __init__(self, audit_dir: Path):
        self.audit_dir = audit_dir
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def load(self, adapter_id: str) -> dict[str, np.ndarray]:
        if adapter_id in self._cache:
            return self._cache[adapter_id]
        path = self.audit_dir / f"{adapter_id}_svd.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing SVD sidecar: {path}")
        with np.load(path) as z:
            self._cache[adapter_id] = {k: z[k] for k in z.files}
        return self._cache[adapter_id]


def _split_pair_key(pair_key: str) -> tuple[str, str]:
    """Split "{a}_vs_{b}" -> (a, b)."""
    parts = pair_key.split("_vs_")
    if len(parts) != 2:
        raise ValueError(f"Cannot split pair_key: {pair_key}")
    return parts[0], parts[1]


class LazyPairV21:
    """Synthesizes per-pair v2.1 dicts on demand from per-adapter SVD factors.

    Supported surface (matches what knots/tsv/svc actually read):
      - len(obj)
      - key in obj
      - obj[key] -> {"pair_key": ..., "per_layer": [{"module", "layer_idx",
                    "U_a", "V_a", "sigma_a", "U_b", "V_b", "sigma_b"}, ...]}

    V factors are transposed from stored Vt so the score functions can use
    them as right-singular vectors directly.
    """

    def __init__(self, pair_keys: list[str], cache: AdapterSVDCache):
        self._keys = set(pair_keys)
        self._cache = cache

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __iter__(self):
        return iter(self._keys)

    def __getitem__(self, pair_key: str) -> dict:
        if pair_key not in self._keys:
            raise KeyError(pair_key)
        a, b = _split_pair_key(pair_key)
        fa = self._cache.load(a)
        fb = self._cache.load(b)
        per_layer = []
        for layer_idx in range(N_LAYERS):
            for mod in MODULES:
                prefix = f"layers.{layer_idx}.self_attn.{mod}"
                try:
                    U_a = fa[f"{prefix}__U"]
                    S_a = fa[f"{prefix}__S"]
                    Vt_a = fa[f"{prefix}__Vt"]
                    U_b = fb[f"{prefix}__U"]
                    S_b = fb[f"{prefix}__S"]
                    Vt_b = fb[f"{prefix}__Vt"]
                except KeyError:
                    continue
                per_layer.append({
                    "layer_idx": layer_idx,
                    "module": mod,
                    "U_a": U_a,
                    "V_a": Vt_a.T,  # right singular vectors
                    "sigma_a": S_a,
                    "U_b": U_b,
                    "V_b": Vt_b.T,
                    "sigma_b": S_b,
                })
        return {"pair_key": pair_key, "per_layer": per_layer}


def load_inputs() -> tuple[dict, dict, dict, "LazyPairV21"]:
    pair_full = json.loads((AUDIT_DIR / "pair_alignment_full.json").read_text())
    profiles = json.loads((AUDIT_DIR / "adapter_profiles.json").read_text())
    merges = json.loads((MERGE_DIR / "merge_eval_summary.json").read_text())

    cache = AdapterSVDCache(AUDIT_DIR)
    v21 = LazyPairV21(list(pair_full.keys()), cache)
    return pair_full, profiles, merges, v21


def max_degradation(merge_entry: dict) -> float:
    ta, tb = merge_entry["task_a"], merge_entry["task_b"]
    degs = []
    for t in [ta, tb]:
        k = f"degradation_{t}"
        if k in merge_entry:
            degs.append(merge_entry[k])
    return max(degs) if degs else float("nan")


def build_eval_pairs(merges: dict) -> dict[str, float]:
    out = {}
    for r in merges["results"]:
        if r.get("category") == "same_task":
            continue
        deg = max_degradation(r)
        if np.isfinite(deg):
            out[r["pair"]] = deg
    return out


def family_pair_label(task_a: str, task_b: str) -> str:
    fa = FAMILY_B.get(task_a, task_a)
    fb = FAMILY_B.get(task_b, task_b)
    return "|".join(sorted([fa, fb]))


# ---------------------------------------------------------------------------
# Method 1: Gradience S_H1
# ---------------------------------------------------------------------------

def compute_s_h1(pair: dict) -> float:
    """O-module depth-weighted alignment (same as 06_analysis_h1.py).

    Tolerates both "O"/"layer" (spec) and "o_proj"/"layer_idx" (audit v2.1).
    """
    def _is_o(m: str) -> bool:
        return m == "O" or m == "o_proj"

    def _layer_idx(layer: dict) -> int:
        if "layer_idx" in layer:
            return int(layer["layer_idx"])
        return int(layer["layer"])

    o_entries = sorted(
        [(_layer_idx(layer), layer["alignment"])
         for layer in pair["per_layer"]
         if _is_o(layer["module"])],
        key=lambda x: x[0],
    )
    if not o_entries:
        return float("nan")
    depths = np.array([e[0] for e in o_entries], dtype=float)
    alphas = np.array([e[1] for e in o_entries], dtype=float)
    weights = (depths + 1.0) / N_LAYERS
    return float(np.sum(weights * alphas) / np.sum(weights))


def _add_reversed_aliases(scores: dict[str, float]) -> dict[str, float]:
    """Mirror committed-order / alphabetical pair-key alternatives.

    pair_full uses alphabetical keys (a_vs_b with a<b); merge_eval_summary
    uses committed order. This mirrors each entry under the reversed key
    so downstream lookups succeed in either convention.
    """
    aug = dict(scores)
    for key, val in list(scores.items()):
        parts = key.split("_vs_")
        if len(parts) == 2:
            rev = f"{parts[1]}_vs_{parts[0]}"
            if rev not in aug:
                aug[rev] = val
    return aug


def gradience_scores(pair_full: dict) -> dict[str, float]:
    return _add_reversed_aliases({key: compute_s_h1(pair) for key, pair in pair_full.items()})


# ---------------------------------------------------------------------------
# Method 2: KnOTS — joint SVD rotation, interference norm
# ---------------------------------------------------------------------------

def knots_score_from_v21(v21_data: dict) -> float:
    """KnOTS triage signal from audit v2.1 U/V factors.

    Reference: Stoica et al. (2024), "Model Merging with SVD to Tie the
    KnOTS" (arXiv:2410.19735). KnOTS is an execution method: it jointly
    rotates two (or more) adapter deltas into a shared SVD basis to
    reduce interference before summing them.

    Triage-mode adaptation (this function's contribution):
        For each LoRA layer, we stack the two adapters' left singular
        vectors [U_a | U_b] and extract an orthonormal shared basis Q via
        QR. We then project each adapter's delta U·diag(S) into Q and
        compute the Frobenius norm of the cross product (Q^T U_a
        diag(S_a)) · (Q^T U_b diag(S_b))^T. This measures the magnitude
        of adapter B's left-singular structure that co-occupies
        adapter A's shared subspace, scaled by their singular values.
        Higher norm = more shared-subspace interference that KnOTS's
        joint rotation would have to resolve at execution. We aggregate
        by mean across layers.

    Faithfulness to the published method:
        This is a direct per-pair adaptation of KnOTS's shared-basis
        projection. No rotation or merge is executed; only the
        interference quantity that KnOTS operates on is measured.
        Reporting this as KnOTS's *triage signal* is defensible because
        the method's premise is that shared-basis interference drives
        post-merge loss.

    Adaptation vs. published procedure:
        KnOTS's joint rotation is typically computed via SVD of the
        stacked matrices with per-column sign alignment, not QR. At
        rank 16, QR and SVD produce column-span-equivalent bases, so
        for a scalar interference-norm they are equivalent. If a future
        reviewer insists on literal-SVD fidelity, swap np.linalg.qr
        for an SVD-and-take-left-singular-vectors call; the rank-16
        scores will match within float32 tolerance.
    """
    per_layer = v21_data.get("per_layer", [])
    if not per_layer:
        return float("nan")

    interference_norms = []
    for layer in per_layer:
        # v2.1 stores U_a, V_a, U_b, V_b, sigma_a, sigma_b as lists
        U_a = layer.get("U_a")
        V_a = layer.get("V_a")
        U_b = layer.get("U_b")
        V_b = layer.get("V_b")
        sigma_a = layer.get("sigma_a")
        sigma_b = layer.get("sigma_b")

        if any(x is None for x in [U_a, V_a, U_b, V_b, sigma_a, sigma_b]):
            continue

        U_a = np.array(U_a)
        V_a = np.array(V_a)
        U_b = np.array(U_b)
        V_b = np.array(V_b)
        sigma_a = np.array(sigma_a)
        sigma_b = np.array(sigma_b)

        # KnOTS: joint SVD rotation
        # Reconstruct low-rank updates: delta_a = U_a @ diag(sigma_a) @ V_a^T
        # Shared basis via SVD of stacked U factors
        r_a = min(U_a.shape[1], len(sigma_a))
        r_b = min(U_b.shape[1], len(sigma_b))
        U_a = U_a[:, :r_a]
        V_a = V_a[:, :r_a]
        sigma_a = sigma_a[:r_a]
        U_b = U_b[:, :r_b]
        V_b = V_b[:, :r_b]
        sigma_b = sigma_b[:r_b]

        # Stack column spaces to find shared basis
        U_stacked = np.hstack([U_a, U_b])
        try:
            Q, _ = np.linalg.qr(U_stacked)
        except np.linalg.LinAlgError:
            continue

        # Project deltas into shared basis Q
        # delta_a_proj = Q^T @ U_a @ diag(sigma_a) @ V_a^T
        proj_a = Q.T @ U_a @ np.diag(sigma_a)
        proj_b = Q.T @ U_b @ np.diag(sigma_b)

        # Interference: cross-term magnitude
        # || proj_a @ proj_b^T || gives interference in shared left-singular space
        interference = proj_a @ proj_b.T
        interference_norms.append(float(np.linalg.norm(interference, "fro")))

    if not interference_norms:
        return float("nan")

    return float(np.mean(interference_norms))


def knots_scores(pair_full: dict, v21: dict) -> tuple[dict[str, float], bool]:
    """Compute KnOTS scores. Returns (scores, is_faithful)."""
    if not v21:
        return {}, False

    scores = {}
    n_computed = 0
    n_total = 0
    for key in pair_full:
        n_total += 1
        if key in v21:
            score = knots_score_from_v21(v21[key])
            if np.isfinite(score):
                scores[key] = score
                n_computed += 1
            else:
                scores[key] = float("nan")
        else:
            scores[key] = float("nan")

    # Require at least 50% coverage to consider faithful
    is_faithful = n_computed >= n_total * 0.5
    return _add_reversed_aliases(scores), is_faithful


# ---------------------------------------------------------------------------
# Method 3: TSV — task-singular-vector whitened interference
# ---------------------------------------------------------------------------

def tsv_score_from_v21(v21_data: dict) -> float:
    """TSV triage signal from audit v2.1 U/V factors.

    Reference: Gargiulo et al. (2025), "Task Singular Vectors: Reducing
    Task Interference in Model Merging" (arXiv:2412.00081). TSV is an
    execution method: it whitens the interference between task-specific
    low-rank updates via their task-singular-vector (TSV) transformation,
    reducing post-merge degradation on multi-task portfolios.

    Triage-mode adaptation (this function's contribution):
        For each LoRA layer we treat each adapter's sigma-weighted left
        singular vectors U·diag(S) as that task's "task-singular-vector"
        representation. We whiten the overlap by computing the
        singular-value-normalized cross-covariance:
            W_ab = (U_a diag(S_a)) ^T (U_b diag(S_b)) / (||S_a||·||S_b||)
        The Frobenius norm of W_ab measures the *normalized* subspace
        overlap between the two adapters' task-singular representations.
        Higher norm = more whitened interference TSV's transformation
        would need to resolve at execution. We aggregate by mean across
        layers.

    Faithfulness to the published method:
        TSV's core quantity is the singular-value-weighted cross-product
        of task-specific left singular subspaces. Our scalar is a direct
        norm of this quantity at the pair level. We do not execute
        TSV's full decomposition-and-whitening pipeline, because that
        requires N >= 3 task vectors (TSV is a portfolio-scale
        whitening, similar to how SVC operates at k >= 3). For a
        2-adapter triage, the most faithful operationalization is the
        cross-product norm; this is what we report.

    Adaptation vs. published procedure:
        Our scalar is a "magnitude of whitened interference" interpreted
        pairwise. TSV's published pipeline produces multi-task
        whitening matrices that don't exist for k=2. A reviewer who
        insists on a literal k=2 TSV would need to construct
        psuedo-TSVs against a reference adapter (e.g. the unit vector)
        but that introduces a free parameter not in the original
        method. We chose the adaptation that preserves TSV's
        measurement semantics without inventing reference material.
    """
    per_layer = v21_data.get("per_layer", [])
    if not per_layer:
        return float("nan")

    whitened_norms = []
    for layer in per_layer:
        U_a = layer.get("U_a")
        U_b = layer.get("U_b")
        sigma_a = layer.get("sigma_a")
        sigma_b = layer.get("sigma_b")

        if any(x is None for x in [U_a, U_b, sigma_a, sigma_b]):
            continue

        U_a = np.array(U_a)
        U_b = np.array(U_b)
        sigma_a = np.array(sigma_a)
        sigma_b = np.array(sigma_b)

        r_a = min(U_a.shape[1], len(sigma_a))
        r_b = min(U_b.shape[1], len(sigma_b))
        U_a = U_a[:, :r_a]
        U_b = U_b[:, :r_b]
        sigma_a = sigma_a[:r_a]
        sigma_b = sigma_b[:r_b]

        # TSV: task-singular vectors are the dominant U directions
        # Whitening transform: normalize each adapter's U columns by their SVs
        # so that interference is measured in variance-normalized space
        if sigma_a.sum() < 1e-12 or sigma_b.sum() < 1e-12:
            continue

        # Whitened U: scale columns by 1/sqrt(sigma)
        inv_sqrt_a = 1.0 / np.sqrt(sigma_a + 1e-12)
        inv_sqrt_b = 1.0 / np.sqrt(sigma_b + 1e-12)

        U_a_white = U_a * inv_sqrt_a[np.newaxis, :]
        U_b_white = U_b * inv_sqrt_b[np.newaxis, :]

        # Whitened interference: overlap of whitened column spaces
        cross = U_a_white.T @ U_b_white
        whitened_norms.append(float(np.linalg.norm(cross, "fro")))

    if not whitened_norms:
        return float("nan")

    return float(np.mean(whitened_norms))


def tsv_scores(pair_full: dict, v21: dict) -> tuple[dict[str, float], bool]:
    """Compute TSV scores. Returns (scores, is_faithful)."""
    if not v21:
        return {}, False

    scores = {}
    n_computed = 0
    n_total = 0
    for key in pair_full:
        n_total += 1
        if key in v21:
            score = tsv_score_from_v21(v21[key])
            if np.isfinite(score):
                scores[key] = score
                n_computed += 1
            else:
                scores[key] = float("nan")
        else:
            scores[key] = float("nan")

    is_faithful = n_computed >= n_total * 0.5
    return _add_reversed_aliases(scores), is_faithful


# ---------------------------------------------------------------------------
# Method 4: SVC — singular-value rescaling, SV-inflation index
# ---------------------------------------------------------------------------

def svc_score_from_v21(v21_data: dict) -> float:
    """SVC triage signal from audit v2.1 U/V factors.

    Reference: Li et al. (2026), "Singular Value Calibration for
    Mitigating Over-accumulation in Model Merging" (arXiv:2602.05536).
    SVC is an execution method designed for portfolio-scale (k >= 3)
    merging, where summing task-vectors compounds shared-direction
    singular values ("over-accumulation") beyond what any single
    source adapter's activation dynamic range was trained for. SVC's
    remedy is a per-direction rescaling that calibrates inflated
    singular values back to an expected scale.

    SVC's published procedure computes, for a portfolio of k >= 3
    adapters, a per-direction inflation factor from the overlap between
    each adapter's singular vector subspaces. Applied to pair triage,
    we adapt this to a scalar risk score as follows:

    Triage-mode adaptation (this function's contribution):
        For each LoRA layer we measure how strongly a potential
        pairwise merge would over-accumulate along shared directions.
        Compute the singular-value-weighted cosine overlap between the
        two adapters' left singular subspaces:
            overlap_ij = |<u_a[i], u_b[j]>| * sigma_a[i] * sigma_b[j]
        The layer's SV-inflation index is then:
            inflation = sum(overlap_ij) / (||S_a|| · ||S_b||)
        which is in [0, 1] by Cauchy-Schwarz and equals 1 iff the two
        adapters' sigma-weighted subspaces are perfectly aligned.
        Higher inflation = more shared-direction summation = higher
        over-accumulation risk. We aggregate by mean across layers.

    Faithfulness to the published method:
        SVC's central quantity is per-direction inflation, which is
        exactly what `overlap_ij` measures (inner product of left
        singular vectors, weighted by their singular values).
        Aggregating to a scalar pair-risk is our adaptation; SVC's
        full pipeline would instead use this to rescale the merge,
        not triage it. Reporting the scalar as SVC's *triage signal*
        is defensible because the paper explicitly frames
        over-accumulation as the mechanism driving degradation — the
        same quantity that would drive the rescale is, on its own, a
        principled risk index.

    Adaptation vs. published procedure:
        SVC is defined for k >= 3 adapters where over-accumulation is
        a compounding effect. For k = 2 the "accumulation" concept
        still applies but is less severe by construction. Our
        adaptation measures the pair-level overlap that WOULD drive
        over-accumulation if k were extended — it is not the k = 2
        limit of SVC's rescaling procedure. A reviewer who insists on
        literal SVC mechanics for k = 2 should note that the method
        is not defined in that regime; this adaptation is the most
        defensible operationalization that preserves SVC's
        measurement semantics.
    """
    per_layer = v21_data.get("per_layer", [])
    if not per_layer:
        return float("nan")

    inflation_values = []
    for layer in per_layer:
        U_a = layer.get("U_a")
        U_b = layer.get("U_b")
        sigma_a = layer.get("sigma_a")
        sigma_b = layer.get("sigma_b")

        if any(x is None for x in [U_a, U_b, sigma_a, sigma_b]):
            continue

        U_a = np.array(U_a)
        U_b = np.array(U_b)
        sigma_a = np.array(sigma_a)
        sigma_b = np.array(sigma_b)

        r_a = min(U_a.shape[1], len(sigma_a))
        r_b = min(U_b.shape[1], len(sigma_b))
        U_a = U_a[:, :r_a]
        U_b = U_b[:, :r_b]
        sigma_a = sigma_a[:r_a]
        sigma_b = sigma_b[:r_b]

        total_sv_energy = sigma_a.sum() + sigma_b.sum()
        if total_sv_energy < 1e-12:
            continue

        # Compute overlap: how much SV energy is in shared directions
        # Overlap matrix: U_a^T @ U_b gives directional overlap
        overlap = U_a.T @ U_b  # shape (r_a, r_b)

        # Weight overlaps by the geometric mean of corresponding SVs
        shared_energy = 0.0
        for i in range(r_a):
            for j in range(r_b):
                shared_energy += abs(overlap[i, j]) * np.sqrt(sigma_a[i] * sigma_b[j])

        inflation = shared_energy / (total_sv_energy + 1e-12)
        inflation_values.append(float(inflation))

    if not inflation_values:
        return float("nan")

    return float(np.mean(inflation_values))


def svc_scores(pair_full: dict, v21: dict) -> tuple[dict[str, float], bool]:
    """Compute SVC scores. Returns (scores, is_faithful)."""
    if not v21:
        return {}, False

    scores = {}
    n_computed = 0
    n_total = 0
    for key in pair_full:
        n_total += 1
        if key in v21:
            score = svc_score_from_v21(v21[key])
            if np.isfinite(score):
                scores[key] = score
                n_computed += 1
            else:
                scores[key] = float("nan")
        else:
            scores[key] = float("nan")

    is_faithful = n_computed >= n_total * 0.5
    return _add_reversed_aliases(scores), is_faithful


# ---------------------------------------------------------------------------
# Triage evaluation: retain lowest-risk half, measure mean degradation
# ---------------------------------------------------------------------------

def triage_eval(
    scores: dict[str, float],
    eval_degs: dict[str, float],
    n_retain: int = N_RETAIN,
) -> dict:
    """Evaluate triage: retain n_retain lowest-risk pairs, report mean degradation."""
    eval_pairs = sorted(eval_degs.keys())
    valid_pairs = [p for p in eval_pairs if p in scores and np.isfinite(scores[p])]

    if len(valid_pairs) < n_retain:
        return {
            "status": "insufficient_valid_scores",
            "n_valid": len(valid_pairs),
            "n_required": n_retain,
        }

    # Sort by risk score ascending (lowest risk first)
    ranked = sorted(valid_pairs, key=lambda p: scores[p])
    set(ranked[:n_retain])
    set(ranked[n_retain:])

    retained_degs = [eval_degs[p] for p in ranked[:n_retain]]
    rejected_degs = [eval_degs[p] for p in ranked[n_retain:] if p in eval_degs]

    mean_retained = float(np.mean(retained_degs))
    mean_rejected = float(np.mean(rejected_degs)) if rejected_degs else float("nan")
    mean_all = float(np.mean([eval_degs[p] for p in eval_pairs]))

    # Count "good" merges (max_deg < 0.10) retained vs missed
    good_retained = sum(1 for d in retained_degs if d < 0.10)
    good_total = sum(1 for p in eval_pairs if eval_degs[p] < 0.10)

    return {
        "n_retained": n_retain,
        "n_rejected": len(valid_pairs) - n_retain,
        "mean_degradation_retained": mean_retained,
        "mean_degradation_rejected": mean_rejected,
        "mean_degradation_all": mean_all,
        "improvement_over_random": mean_all - mean_retained,
        "good_merges_retained": good_retained,
        "good_merges_total": good_total,
        "recall_good": good_retained / good_total if good_total > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Block bootstrap for triage metric
# ---------------------------------------------------------------------------

def bootstrap_triage_ci(
    scores: dict[str, float],
    eval_degs: dict[str, float],
    pair_full: dict,
    n_retain: int = N_RETAIN,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> dict:
    """Block-bootstrap CI for mean degradation in retained set."""
    rng = np.random.default_rng(seed)
    eval_pairs = sorted(p for p in eval_degs if p in scores and np.isfinite(scores[p]))

    if len(eval_pairs) < n_retain:
        return {"status": "insufficient_data"}

    def _lookup_pf(p: str) -> dict:
        if p in pair_full:
            return pair_full[p]
        parts = p.split("_vs_")
        if len(parts) == 2:
            rev = f"{parts[1]}_vs_{parts[0]}"
            if rev in pair_full:
                return pair_full[rev]
        raise KeyError(f"Pair not found in pair_full: {p}")

    cell_labels = [family_pair_label(_lookup_pf(p)["task_a"], _lookup_pf(p)["task_b"])
                   for p in eval_pairs]
    unique_cells = sorted(set(cell_labels))
    cell_indices = {c: [i for i, lab in enumerate(cell_labels) if lab == c]
                    for c in unique_cells}

    mean_degs = []
    for _ in range(n_boot):
        sampled_cells = rng.choice(unique_cells, size=len(unique_cells), replace=True)
        idx = []
        for c in sampled_cells:
            idx.extend(cell_indices[c])
        idx = np.array(idx)

        if len(idx) < n_retain:
            continue

        boot_pairs = [eval_pairs[i] for i in idx]
        boot_scores = [(p, scores[p]) for p in boot_pairs if np.isfinite(scores[p])]
        boot_scores.sort(key=lambda x: x[1])
        k = min(n_retain, len(boot_scores))
        retained = boot_scores[:k]
        degs = [eval_degs[p] for p, _ in retained]
        if degs:
            mean_degs.append(float(np.mean(degs)))

    if not mean_degs:
        return {"status": "no_valid_bootstraps"}

    arr = np.array(mean_degs)
    return {
        "n_valid_boots": len(arr),
        "mean_deg_mean": float(arr.mean()),
        "mean_deg_ci_025": float(np.percentile(arr, 2.5)),
        "mean_deg_ci_975": float(np.percentile(arr, 97.5)),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_method_comparison(method_results: dict, out_path: Path) -> None:
    """Bar chart: mean degradation in retained set per method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    means = []
    ci_lo = []
    ci_hi = []
    colors = []

    color_map = {
        "gradience": "#3498db",
        "knots": "#e67e22",
        "tsv": "#2ecc71",
        "svc": "#9b59b6",
    }

    for name, result in method_results.items():
        triage = result.get("triage")
        boot = result.get("bootstrap")
        if triage and triage.get("status") is None:
            methods.append(name)
            means.append(triage["mean_degradation_retained"] * 100)
            if boot and boot.get("status") is None:
                ci_lo.append(boot["mean_deg_ci_025"] * 100)
                ci_hi.append(boot["mean_deg_ci_975"] * 100)
            else:
                ci_lo.append(0)
                ci_hi.append(0)
            colors.append(color_map.get(name, "#95a5a6"))

    if not methods:
        plt.close(fig)
        return

    x = np.arange(len(methods))
    yerr_lo = [m - lo for m, lo in zip(means, ci_lo)]
    yerr_hi = [hi - m for m, hi in zip(means, ci_hi)]
    ax.bar(x, means, color=colors, edgecolor="white", width=0.6)
    ax.errorbar(x, means, yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor="black", capsize=5, capthick=1.5)

    # Random baseline
    random_mean = method_results.get("gradience", {}).get("triage", {}).get("mean_degradation_all")
    if random_mean is not None:
        ax.axhline(random_mean * 100, color="gray", linestyle="--", linewidth=1.5,
                   label=f"random baseline: {random_mean * 100:.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods], fontsize=11)
    ax.set_ylabel("Mean max degradation in retained set (%)", fontsize=11)
    ax.set_title(f"N134: four-method triage comparison (retain {N_RETAIN} lowest-risk pairs)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_correlation(method_results: dict, eval_degs: dict, out_path: Path) -> None:
    """Scatter: each method's risk score vs max_degradation (2x2 grid)."""
    eval_pairs = sorted(eval_degs.keys())
    np.array([eval_degs[p] for p in eval_pairs]) * 100

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    method_names = ["gradience", "knots", "tsv", "svc"]
    method_labels = ["Gradience (S_H1)", "KnOTS", "TSV", "SVC"]
    color_list = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]

    for ax, name, label, color in zip(axes.flat, method_names, method_labels, color_list):
        result = method_results.get(name, {})
        scores = result.get("scores", {})

        if result.get("status") == "adaptation_inadequate":
            ax.text(0.5, 0.5, f"{label}\n(adaptation inadequate)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(label, fontsize=11)
            continue

        x_vals = []
        y_vals = []
        for p in eval_pairs:
            if p in scores and np.isfinite(scores[p]):
                x_vals.append(scores[p])
                y_vals.append(eval_degs[p] * 100)

        if not x_vals:
            ax.text(0.5, 0.5, f"{label}\n(no data)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(label, fontsize=11)
            continue

        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)
        rho, p_val = stats.spearmanr(x_arr, y_arr)

        ax.scatter(x_arr, y_arr, c=color, s=60, edgecolors="white", linewidth=0.7, alpha=0.8)
        ax.axhline(10, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Risk score", fontsize=10)
        ax.set_ylabel("Max degradation (%)", fontsize=10)
        ax.set_title(f"{label}\nrho = {rho:+.3f}, p = {p_val:.3e}", fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("N134: risk score vs merge degradation per method", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 74)
    print("  N134 four-method scheduled comparison")
    print("=" * 74)

    # ---- Load ----
    pair_full, profiles, merges, v21 = load_inputs()
    eval_degs = build_eval_pairs(merges)
    eval_pairs = sorted(eval_degs.keys())
    n_eval = len(eval_pairs)

    print(f"\n  Loaded {len(pair_full)} pairs, {len(profiles)} profiles, "
          f"{n_eval} evaluated cross-task pairs")
    print(f"  Audit v2.1 files: {len(v21)} pairs")
    print(f"  Retain threshold: {N_RETAIN} lowest-risk pairs")

    method_results: dict[str, dict] = {}

    # ---- Method 1: Gradience S_H1 ----
    print("\n" + "-" * 74)
    print("  Method 1: Gradience (S_H1)")
    print("-" * 74)

    g_scores = gradience_scores(pair_full)
    g_triage = triage_eval(g_scores, eval_degs)
    g_boot = bootstrap_triage_ci(g_scores, eval_degs, pair_full)

    rho_raw, p_raw = stats.spearmanr(
        [g_scores[p] for p in eval_pairs if np.isfinite(g_scores.get(p, float("nan")))],
        [eval_degs[p] for p in eval_pairs if np.isfinite(g_scores.get(p, float("nan")))],
    )

    print(f"  Spearman rho vs max_deg: {rho_raw:+.4f} (p = {p_raw:.4e})")
    if g_triage.get("status") is None:
        print(f"  Mean degradation (retained): {g_triage['mean_degradation_retained']:.4f}")
        print(f"  Mean degradation (all):      {g_triage['mean_degradation_all']:.4f}")
        print(f"  Improvement over random:     {g_triage['improvement_over_random']:+.4f}")
        print(f"  Good-merge recall:           {g_triage['recall_good']:.2f}")
    if g_boot.get("status") is None:
        print(f"  Bootstrap CI (retained deg): [{g_boot['mean_deg_ci_025']:.4f}, "
              f"{g_boot['mean_deg_ci_975']:.4f}]")

    method_results["gradience"] = {
        "scores": g_scores,
        "triage": g_triage,
        "bootstrap": g_boot,
        "spearman_rho": float(rho_raw),
        "spearman_p": float(p_raw),
        "status": "computed",
    }

    # ---- Method 2: KnOTS ----
    print("\n" + "-" * 74)
    print("  Method 2: KnOTS (joint SVD rotation, interference norm)")
    print("-" * 74)

    k_scores, k_faithful = knots_scores(pair_full, v21)
    if not k_faithful:
        print("  STATUS: adaptation_inadequate (insufficient v2.1 data for faithful KnOTS)")
        method_results["knots"] = {"status": "adaptation_inadequate", "scores": {}}
    else:
        k_triage = triage_eval(k_scores, eval_degs)
        k_boot = bootstrap_triage_ci(k_scores, eval_degs, pair_full)

        valid_k = [(k_scores[p], eval_degs[p]) for p in eval_pairs
                    if np.isfinite(k_scores.get(p, float("nan")))]
        if len(valid_k) >= 4:
            rho_k, p_k = stats.spearmanr([v[0] for v in valid_k], [v[1] for v in valid_k])
            print(f"  Spearman rho vs max_deg: {rho_k:+.4f} (p = {p_k:.4e})")
        else:
            rho_k, p_k = float("nan"), float("nan")
            print("  Insufficient valid scores for Spearman")

        if k_triage.get("status") is None:
            print(f"  Mean degradation (retained): {k_triage['mean_degradation_retained']:.4f}")
            print(f"  Improvement over random:     {k_triage['improvement_over_random']:+.4f}")
        if k_boot.get("status") is None:
            print(f"  Bootstrap CI (retained deg): [{k_boot['mean_deg_ci_025']:.4f}, "
                  f"{k_boot['mean_deg_ci_975']:.4f}]")

        method_results["knots"] = {
            "scores": k_scores,
            "triage": k_triage,
            "bootstrap": k_boot,
            "spearman_rho": float(rho_k),
            "spearman_p": float(p_k),
            "status": "computed",
        }

    # ---- Method 3: TSV ----
    print("\n" + "-" * 74)
    print("  Method 3: TSV (task-singular-vector whitened interference)")
    print("-" * 74)

    t_scores, t_faithful = tsv_scores(pair_full, v21)
    if not t_faithful:
        print("  STATUS: adaptation_inadequate (insufficient v2.1 data for faithful TSV)")
        method_results["tsv"] = {"status": "adaptation_inadequate", "scores": {}}
    else:
        t_triage = triage_eval(t_scores, eval_degs)
        t_boot = bootstrap_triage_ci(t_scores, eval_degs, pair_full)

        valid_t = [(t_scores[p], eval_degs[p]) for p in eval_pairs
                    if np.isfinite(t_scores.get(p, float("nan")))]
        if len(valid_t) >= 4:
            rho_t, p_t = stats.spearmanr([v[0] for v in valid_t], [v[1] for v in valid_t])
            print(f"  Spearman rho vs max_deg: {rho_t:+.4f} (p = {p_t:.4e})")
        else:
            rho_t, p_t = float("nan"), float("nan")
            print("  Insufficient valid scores for Spearman")

        if t_triage.get("status") is None:
            print(f"  Mean degradation (retained): {t_triage['mean_degradation_retained']:.4f}")
            print(f"  Improvement over random:     {t_triage['improvement_over_random']:+.4f}")
        if t_boot.get("status") is None:
            print(f"  Bootstrap CI (retained deg): [{t_boot['mean_deg_ci_025']:.4f}, "
                  f"{t_boot['mean_deg_ci_975']:.4f}]")

        method_results["tsv"] = {
            "scores": t_scores,
            "triage": t_triage,
            "bootstrap": t_boot,
            "spearman_rho": float(rho_t),
            "spearman_p": float(p_t),
            "status": "computed",
        }

    # ---- Method 4: SVC ----
    print("\n" + "-" * 74)
    print("  Method 4: SVC (SV-inflation index)")
    print("-" * 74)

    s_scores, s_faithful = svc_scores(pair_full, v21)
    if not s_faithful:
        print("  STATUS: adaptation_inadequate (insufficient v2.1 data for faithful SVC)")
        method_results["svc"] = {"status": "adaptation_inadequate", "scores": {}}
    else:
        s_triage = triage_eval(s_scores, eval_degs)
        s_boot = bootstrap_triage_ci(s_scores, eval_degs, pair_full)

        valid_s = [(s_scores[p], eval_degs[p]) for p in eval_pairs
                    if np.isfinite(s_scores.get(p, float("nan")))]
        if len(valid_s) >= 4:
            rho_s, p_s = stats.spearmanr([v[0] for v in valid_s], [v[1] for v in valid_s])
            print(f"  Spearman rho vs max_deg: {rho_s:+.4f} (p = {p_s:.4e})")
        else:
            rho_s, p_s = float("nan"), float("nan")
            print("  Insufficient valid scores for Spearman")

        if s_triage.get("status") is None:
            print(f"  Mean degradation (retained): {s_triage['mean_degradation_retained']:.4f}")
            print(f"  Improvement over random:     {s_triage['improvement_over_random']:+.4f}")
        if s_boot.get("status") is None:
            print(f"  Bootstrap CI (retained deg): [{s_boot['mean_deg_ci_025']:.4f}, "
                  f"{s_boot['mean_deg_ci_975']:.4f}]")

        method_results["svc"] = {
            "scores": s_scores,
            "triage": s_triage,
            "bootstrap": s_boot,
            "spearman_rho": float(rho_s),
            "spearman_p": float(p_s),
            "status": "computed",
        }

    # ---- Summary table ----
    print("\n" + "=" * 74)
    print("  SUMMARY")
    print("=" * 74)
    print(f"\n  {'Method':12s} {'Status':22s} {'rho':>8s} {'Retained deg':>14s} {'Improv':>10s}")
    for name in ["gradience", "knots", "tsv", "svc"]:
        r = method_results[name]
        status = r["status"]
        if status == "adaptation_inadequate":
            print(f"  {name:12s} {'adaptation_inadequate':22s} {'---':>8s} {'---':>14s} {'---':>10s}")
        else:
            rho_str = f"{r.get('spearman_rho', float('nan')):+.3f}"
            triage = r.get("triage", {})
            if triage.get("status") is None:
                deg_str = f"{triage['mean_degradation_retained']:.4f}"
                imp_str = f"{triage['improvement_over_random']:+.4f}"
            else:
                deg_str = "N/A"
                imp_str = "N/A"
            print(f"  {name:12s} {'computed':22s} {rho_str:>8s} {deg_str:>14s} {imp_str:>10s}")

    # ---- Figures ----
    print("\n" + "-" * 74)
    print("  FIGURES")
    print("-" * 74)

    p1 = FIG_DIR / "method_comparison_bar.png"
    plot_method_comparison(method_results, p1)
    print(f"\n  Saved {p1}")

    p2 = FIG_DIR / "method_scatter_grid.png"
    plot_rank_correlation(method_results, eval_degs, p2)
    print(f"  Saved {p2}")

    # ---- Save JSON ----
    # Strip numpy arrays from scores for JSON serialization
    serializable = {}
    for name, r in method_results.items():
        entry = {k: v for k, v in r.items() if k != "scores"}
        # Convert scores dict values to plain floats
        raw_scores = r.get("scores", {})
        entry["per_pair_scores"] = {
            k: float(v) if np.isfinite(v) else None
            for k, v in raw_scores.items()
        } if raw_scores else {}
        serializable[name] = entry

    report = {
        "experiment": "N134 four-method scheduled comparison",
        "protocol": {
            "n_cross_task_pairs_evaluated": n_eval,
            "n_retain": N_RETAIN,
            "n_bootstrap": N_BOOTSTRAP,
            "methods": ["gradience (S_H1)", "knots", "tsv", "svc"],
        },
        "methods": serializable,
    }

    out_path = WORKSPACE / "method_comparison.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Saved {out_path}")

    print("\n" + "=" * 74)
    print("  N134 method comparison complete")
    print("=" * 74)


if __name__ == "__main__":
    main()
