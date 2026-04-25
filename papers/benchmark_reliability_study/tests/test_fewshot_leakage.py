"""Tests for 02_draw_fewshot_examples.py — SPEC_CPU_v0_2 §8.3.

Six tests cover:
    1. Clean (no-leakage) end-to-end draw.
    2. Synthetic leakage triggers FewshotLeakageError + exit code 3.
    3. Determinism: same inputs → byte-identical outputs.
    4. 0-shot benchmark emits a single placeholder row, no draws.
    5. Subjects (e.g. MMLU panel) are drawn separately per subject.
    6. CSV schema and column order match SPEC.

Tests 1 and 5 call ``main()`` directly with an injected fixture loader.
Tests 2, 3, 4, and 6 invoke the CLI via subprocess. Tests 2 and 3 use a
small harness shim (``_invoke_with_fixture_loader``) that reconstructs
the fixture loader inside a child process from a JSON file. Tests 4 and
6 use a 0-shot-only config so the default loader is never called.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PAPER_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PAPER_ROOT / "scripts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCRIPT_PATH = SCRIPTS_DIR / "02_draw_fewshot_examples.py"


# -- Importing the script as a module ---------------------------------------
# The script's filename starts with a digit, which prevents standard
# ``import`` syntax. Use importlib.


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "draw_fewshot_examples", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


draw_module = _load_script_module()
main = draw_module.main
MANIFEST_FIELDNAMES = draw_module.MANIFEST_FIELDNAMES
DRAW_SCRIPT_VERSION = draw_module.DRAW_SCRIPT_VERSION

from gradience_study.config import FewshotLeakageError  # noqa: E402


# -- Fixture data builders --------------------------------------------------


def make_fixture_loader(split_data):
    """Returns a loader function callable as ``loader(hf_dataset, hf_config, split, version_hash)``.

    ``split_data`` is a dict mapping (hf_dataset, split) → list of item dicts.
    """

    def loader(hf_dataset, hf_config, split, version_hash):
        return list(split_data.get((hf_dataset, split), []))

    return loader


def _items_with_ids(ids):
    """Build minimal item dicts whose ``id`` field equals each provided ID."""
    return [{"id": i} for i in ids]


def _items_with_subject(ids, subject):
    """Build item dicts with ``id`` and a ``subject`` field."""
    return [{"id": i, "subject": subject} for i in ids]


# -- Fixture configs --------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def ensure_fixtures_built():
    """Build fixture configs if missing. Idempotent."""
    cfg_dir = FIXTURES_DIR / "configs"
    if not (cfg_dir / "study_config.yaml").exists():
        result = subprocess.run(
            [sys.executable, str(FIXTURES_DIR / "make_fixtures.py")],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Fixture build failed: {result.stderr}"


@pytest.fixture
def paper_root(tmp_path):
    """A minimal paper-root layout with full fixture configs."""
    root = tmp_path / "paper_root"
    (root / "configs").mkdir(parents=True)
    for src in (FIXTURES_DIR / "configs").glob("*.yaml"):
        shutil.copy(src, root / "configs" / src.name)
    (root / "manifests").mkdir()
    (root / "preregistration" / "appendices").mkdir(parents=True)
    return root


def _zero_shot_only_benchmarks_yaml() -> str:
    """A benchmarks.yaml with a single 0-shot primary benchmark.

    Used by tests that only exercise placeholder-row behavior so the
    default loader is never invoked.
    """
    return (
        "benchmarks:\n"
        "  - benchmark_id: test_bench_3\n"
        "    display_name: Test Benchmark 3\n"
        "    tier: primary\n"
        "    hf_dataset: test/bench3\n"
        "    hf_config: null\n"
        "    dataset_version_hash: hash3\n"
        "    task_type: constrained_choice\n"
        "    fewshot_k: 0\n"
        "    seed_facet: false\n"
        "    fewshot_source_split: null\n"
        "    eval_split: validation\n"
        "    scoring_rules: [ll_norm, generate_parse]\n"
        "    item_id_field: id\n"
        "    answer_field: answer\n"
        "    expected_num_items: 10\n"
    )


@pytest.fixture
def paper_root_zero_shot_only(tmp_path):
    """Paper root whose benchmarks.yaml contains only a 0-shot benchmark."""
    root = tmp_path / "paper_root_zs"
    (root / "configs").mkdir(parents=True)
    for src in (FIXTURES_DIR / "configs").glob("*.yaml"):
        if src.name == "benchmarks.yaml":
            continue
        shutil.copy(src, root / "configs" / src.name)
    (root / "configs" / "benchmarks.yaml").write_text(
        _zero_shot_only_benchmarks_yaml(), encoding="utf-8"
    )
    (root / "manifests").mkdir()
    (root / "preregistration" / "appendices").mkdir(parents=True)
    return root


# -- Default fixture loader for the 5 fixture benchmarks --------------------


def _default_split_data(*, leak: bool = False, eval_size: int = 50):
    """Build (hf_dataset, split) → items for fixture benchmarks 1..5.

    bench-1 uses eval_split=test, others use validation. bench-3 is 0-shot
    so its source split is irrelevant. With ``leak=True`` we deliberately
    inject overlap between fewshot and eval IDs for test_bench_2.
    """
    data = {}

    # Bench 1: source=train (size 100), eval=test (size 50). Disjoint.
    data[("test/bench1", "train")] = _items_with_ids(
        [f"b1-train-{i:04d}" for i in range(100)]
    )
    data[("test/bench1", "test")] = _items_with_ids(
        [f"b1-eval-{i:04d}" for i in range(eval_size)]
    )

    # Bench 2: source=train (size 80), eval=validation (size eval_size).
    if leak:
        # Construct overlap deliberately: train and validation share IDs.
        shared_ids = [f"b2-shared-{i:04d}" for i in range(80)]
        data[("test/bench2", "train")] = _items_with_ids(shared_ids)
        data[("test/bench2", "validation")] = _items_with_ids(shared_ids[:eval_size])
    else:
        data[("test/bench2", "train")] = _items_with_ids(
            [f"b2-train-{i:04d}" for i in range(80)]
        )
        data[("test/bench2", "validation")] = _items_with_ids(
            [f"b2-eval-{i:04d}" for i in range(eval_size)]
        )

    # Bench 3 is 0-shot — no source needed, but eval is still loaded by
    # the leakage walker. Leakage walker actually skips benches with no
    # fewshot rows, but we still provide the eval split for safety.
    data[("test/bench3", "validation")] = _items_with_ids(
        [f"b3-eval-{i:04d}" for i in range(eval_size)]
    )

    # Bench 4 and 5: source=train (size 60), eval=validation. Disjoint.
    for n in (4, 5):
        data[(f"test/bench{n}", "train")] = _items_with_ids(
            [f"b{n}-train-{i:04d}" for i in range(60)]
        )
        data[(f"test/bench{n}", "validation")] = _items_with_ids(
            [f"b{n}-eval-{i:04d}" for i in range(eval_size)]
        )

    return data


# -- Subprocess harness for tests that need a custom loader -----------------


_HARNESS_TEMPLATE = """\
import json
import sys
from pathlib import Path

sys.path.insert(0, {paper_root!r})

import importlib.util
spec = importlib.util.spec_from_file_location("dfs", {script_path!r})
dfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dfs)

with open({fixture_json!r}, "r", encoding="utf-8") as f:
    raw = json.load(f)
# raw is a list of [hf_dataset, split, items] triples.
table = {{(entry["hf_dataset"], entry["split"]): entry["items"]
          for entry in raw}}

def loader(hf_dataset, hf_config, split, version_hash):
    return list(table.get((hf_dataset, split), []))

sys.exit(dfs.main(loader=loader))
"""


def _serialize_split_data(split_data) -> list[dict]:
    """Convert (hf_dataset, split) → items dict into a JSON-serializable list."""
    return [
        {"hf_dataset": k[0], "split": k[1], "items": v}
        for k, v in split_data.items()
    ]


def _invoke_with_fixture_loader(
    *, paper_root: Path, split_data, cli_args: list[str], tmp_path: Path
):
    """Run the script via subprocess with an injected fixture loader.

    Writes split_data to a JSON file, generates a small Python harness
    that imports ``main`` and calls it with a loader built from that file,
    and runs the harness with ``sys.executable``.
    """
    fixture_json = tmp_path / "fixture_split_data.json"
    fixture_json.write_text(
        json.dumps(_serialize_split_data(split_data)), encoding="utf-8"
    )
    harness_path = tmp_path / "harness.py"
    harness_src = _HARNESS_TEMPLATE.format(
        paper_root=str(PAPER_ROOT),
        script_path=str(SCRIPT_PATH),
        fixture_json=str(fixture_json),
    )
    harness_path.write_text(harness_src, encoding="utf-8")

    return subprocess.run(
        [sys.executable, str(harness_path), *cli_args],
        capture_output=True,
        text=True,
        cwd=str(paper_root),
    )


# ============================================================================
# Test 1 — clean draw, no leakage
# ============================================================================


def test_no_leakage_passes_clean(paper_root, tmp_path):
    """Disjoint fewshot/eval splits → main() returns 0 and emits both files."""
    out_csv = paper_root / "manifests" / "fewshot_manifest.csv"
    out_lock = paper_root / "preregistration" / "appendices" / "fewshot_draws_LOCKED.json"
    loader = make_fixture_loader(_default_split_data())

    rc = main(
        argv=[
            "--config", str(paper_root / "configs" / "study_config.yaml"),
            "--seeds", "42", "123", "2024",
            "--out", str(out_csv),
            "--lock-out", str(out_lock),
            "--fixed-timestamp", "2026-04-24T00:00:00Z",
        ],
        loader=loader,
    )

    assert rc == 0, "Expected exit code 0 for a clean draw."
    assert out_csv.exists(), "Manifest CSV should be written."
    assert out_lock.exists(), "LOCKED.json should be written."

    with open(out_lock, encoding="utf-8") as f:
        lock = json.load(f)
    assert lock["draw_script_version"] == DRAW_SCRIPT_VERSION
    assert "config_hash" in lock and len(lock["config_hash"]) == 8
    assert lock["seeds"] == [42, 123, 2024]
    # Four benchmarks have fewshot_k > 0 (test_bench_3 is 0-shot).
    assert set(lock["draws"].keys()) == {
        "test_bench_1", "test_bench_2", "test_bench_4", "test_bench_5"
    }
    for bench_id, by_subject in lock["draws"].items():
        assert "null" in by_subject
        assert set(by_subject["null"].keys()) == {"42", "123", "2024"}
        for ids in by_subject["null"].values():
            assert len(ids) == 5  # fewshot_k


# ============================================================================
# Test 2 — synthetic leakage hard-fails with exit code 3
# ============================================================================


def test_leakage_raises_hard_fail(paper_root, tmp_path):
    """Inject overlap between train and validation for bench 2."""
    out_csv = paper_root / "manifests" / "fewshot_manifest.csv"
    out_lock = paper_root / "preregistration" / "appendices" / "fewshot_draws_LOCKED.json"

    # Direct in-process check first: confirm the exception class is right.
    leaky_loader = make_fixture_loader(_default_split_data(leak=True))
    with pytest.raises(FewshotLeakageError):
        # Exit code path swallows the exception; for this assertion we
        # call the leakage-check function directly via the script's API.
        rows, _ = draw_module._build_manifest(
            merged=_load_merged(paper_root),
            seeds=[42],
            loader=leaky_loader,
            draw_timestamp="2026-04-24T00:00:00Z",
        )
        draw_module._validate_no_leakage(rows, _load_merged(paper_root), leaky_loader)

    # Now via subprocess: confirm the script exits with code 3.
    result = _invoke_with_fixture_loader(
        paper_root=paper_root,
        split_data=_default_split_data(leak=True),
        cli_args=[
            "--config", str(paper_root / "configs" / "study_config.yaml"),
            "--seeds", "42", "123", "2024",
            "--out", str(out_csv),
            "--lock-out", str(out_lock),
            "--fixed-timestamp", "2026-04-24T00:00:00Z",
        ],
        tmp_path=tmp_path,
    )
    assert result.returncode == 3, (
        f"Expected exit 3 (leakage); got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "leakage" in result.stderr.lower(), (
        "Leakage message should appear in stderr."
    )


def _load_merged(paper_root: Path):
    """Helper: load merged config for direct API checks inside test 2."""
    from gradience_study.config import load_config
    return load_config(paper_root / "configs" / "study_config.yaml")


# ============================================================================
# Test 3 — determinism: same seeds + config + data → byte-identical outputs
# ============================================================================


def test_deterministic_draws(paper_root, tmp_path):
    """Two runs in different output directories produce identical files."""
    split_data = _default_split_data()

    def run_once(suffix: str):
        out_dir = tmp_path / f"out_{suffix}"
        out_dir.mkdir()
        out_csv = out_dir / "fewshot_manifest.csv"
        out_lock = out_dir / "fewshot_draws_LOCKED.json"
        result = _invoke_with_fixture_loader(
            paper_root=paper_root,
            split_data=split_data,
            cli_args=[
                "--config", str(paper_root / "configs" / "study_config.yaml"),
                "--seeds", "42", "123", "2024",
                "--out", str(out_csv),
                "--lock-out", str(out_lock),
                "--fixed-timestamp", "2026-04-24T00:00:00Z",
            ],
            tmp_path=out_dir,
        )
        assert result.returncode == 0, (
            f"Run {suffix} failed: {result.stderr}"
        )
        return out_csv, out_lock

    csv_a, lock_a = run_once("a")
    csv_b, lock_b = run_once("b")

    def file_sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    assert file_sha(csv_a) == file_sha(csv_b), (
        "Manifest CSV not byte-identical across runs."
    )
    assert file_sha(lock_a) == file_sha(lock_b), (
        "LOCKED.json not byte-identical across runs."
    )


# ============================================================================
# Test 4 — 0-shot benchmark: single placeholder row, no draws populated
# ============================================================================


def test_zero_shot_benchmark(paper_root_zero_shot_only, tmp_path):
    """fewshot_k == 0 → one row with seed_id='none' and empty example_item_id."""
    paper_root = paper_root_zero_shot_only
    out_csv = paper_root / "manifests" / "fewshot_manifest.csv"
    out_lock = paper_root / "preregistration" / "appendices" / "fewshot_draws_LOCKED.json"

    # No loader needed: 0-shot benchmarks don't draw and don't trigger
    # leakage checks for themselves. The default loader's NotImplementedError
    # is never reached.
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--config", str(paper_root / "configs" / "study_config.yaml"),
            "--seeds", "42", "123", "2024",
            "--out", str(out_csv),
            "--lock-out", str(out_lock),
            "--fixed-timestamp", "2026-04-24T00:00:00Z",
        ],
        capture_output=True,
        text=True,
        cwd=str(paper_root),
    )
    assert result.returncode == 0, (
        f"Expected exit 0; got {result.returncode}.\nstderr: {result.stderr}"
    )

    with open(out_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1, f"Expected exactly 1 placeholder row; got {len(rows)}."
    row = rows[0]
    assert row["benchmark_id"] == "test_bench_3"
    assert row["seed_id"] == "none"
    assert row["fewshot_k"] == "0"
    assert row["example_rank"] == "0"
    assert row["example_item_id"] == ""
    assert row["draw_script_version"] == DRAW_SCRIPT_VERSION

    with open(out_lock, encoding="utf-8") as f:
        lock = json.load(f)
    # No draws populated for the 0-shot benchmark.
    assert "test_bench_3" not in lock["draws"], (
        "0-shot benchmark must not appear in LOCKED.json draws map."
    )
    assert lock["draws"] == {}, (
        f"Expected empty draws map for 0-shot-only config; got {lock['draws']}."
    )


# ============================================================================
# Test 5 — MMLU-style subjects drawn separately
# ============================================================================


def test_mmlu_style_subjects_drawn_separately(tmp_path):
    """Subject-bearing benchmark draws fewshot_k items per subject per seed."""
    # Build a paper root with a single subject-bearing benchmark.
    root = tmp_path / "paper_root_mmlu"
    (root / "configs").mkdir(parents=True)
    for src in (FIXTURES_DIR / "configs").glob("*.yaml"):
        if src.name == "benchmarks.yaml":
            continue
        shutil.copy(src, root / "configs" / src.name)
    (root / "manifests").mkdir()
    (root / "preregistration" / "appendices").mkdir(parents=True)

    benchmarks_yaml = (
        "benchmarks:\n"
        "  - benchmark_id: mmlu_panel\n"
        "    display_name: MMLU panel (test fixture)\n"
        "    tier: primary\n"
        "    hf_dataset: test/mmlu\n"
        "    hf_config: null\n"
        "    dataset_version_hash: hash_mmlu\n"
        "    task_type: constrained_choice\n"
        "    fewshot_k: 3\n"
        "    seed_facet: true\n"
        "    fewshot_source_split: dev\n"
        "    eval_split: test\n"
        "    scoring_rules: [ll_norm, generate_parse]\n"
        "    item_id_field: id\n"
        "    answer_field: answer\n"
        "    expected_num_items: 10\n"
        "    subjects:\n"
        "      - anatomy\n"
        "      - astronomy\n"
        "      - philosophy\n"
    )
    (root / "configs" / "benchmarks.yaml").write_text(benchmarks_yaml, encoding="utf-8")

    # Build source items: dev split contains items for all three subjects;
    # eval split disjoint.
    dev_items = []
    for subj in ("anatomy", "astronomy", "philosophy"):
        for i in range(20):
            dev_items.append({"id": f"{subj}-dev-{i:03d}", "subject": subj})
    eval_items = []
    for subj in ("anatomy", "astronomy", "philosophy"):
        for i in range(10):
            eval_items.append({"id": f"{subj}-test-{i:03d}", "subject": subj})

    split_data = {
        ("test/mmlu", "dev"): dev_items,
        ("test/mmlu", "test"): eval_items,
    }
    loader = make_fixture_loader(split_data)

    out_csv = root / "manifests" / "fewshot_manifest.csv"
    out_lock = root / "preregistration" / "appendices" / "fewshot_draws_LOCKED.json"

    rc = main(
        argv=[
            "--config", str(root / "configs" / "study_config.yaml"),
            "--seeds", "42", "123",
            "--out", str(out_csv),
            "--lock-out", str(out_lock),
            "--fixed-timestamp", "2026-04-24T00:00:00Z",
        ],
        loader=loader,
    )
    assert rc == 0

    with open(out_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # 3 subjects × 2 seeds × 3 exemplars = 18 rows.
    assert len(rows) == 18, f"Expected 18 rows; got {len(rows)}."

    # Group by (subject, seed) and verify k=3 each, IDs from that subject only.
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["subject_id"], row["seed_id"])].append(row)
    assert set(grouped.keys()) == {
        ("anatomy", "42"), ("anatomy", "123"),
        ("astronomy", "42"), ("astronomy", "123"),
        ("philosophy", "42"), ("philosophy", "123"),
    }
    for (subj, seed), bucket in grouped.items():
        assert len(bucket) == 3, f"({subj}, {seed}) expected 3 rows; got {len(bucket)}."
        ranks = sorted(int(r["example_rank"]) for r in bucket)
        assert ranks == [0, 1, 2]
        for r in bucket:
            assert r["example_item_id"].startswith(f"{subj}-dev-"), (
                f"Item ID {r['example_item_id']} should belong to subject {subj}."
            )

    # LOCKED.json should have one entry per subject under mmlu_panel.
    with open(out_lock, encoding="utf-8") as f:
        lock = json.load(f)
    assert set(lock["draws"]["mmlu_panel"].keys()) == {
        "anatomy", "astronomy", "philosophy"
    }
    for subj in ("anatomy", "astronomy", "philosophy"):
        per_seed = lock["draws"]["mmlu_panel"][subj]
        assert set(per_seed.keys()) == {"42", "123"}
        for ids in per_seed.values():
            assert len(ids) == 3
            assert all(i.startswith(f"{subj}-dev-") for i in ids)


# ============================================================================
# Test 6 — manifest CSV schema (column order)
# ============================================================================


def test_manifest_schema(paper_root_zero_shot_only):
    """Manifest CSV header equals MANIFEST_FIELDNAMES in order."""
    paper_root = paper_root_zero_shot_only
    out_csv = paper_root / "manifests" / "fewshot_manifest.csv"
    out_lock = paper_root / "preregistration" / "appendices" / "fewshot_draws_LOCKED.json"

    # 0-shot-only config: no loader needed.
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--config", str(paper_root / "configs" / "study_config.yaml"),
            "--seeds", "42",
            "--out", str(out_csv),
            "--lock-out", str(out_lock),
            "--fixed-timestamp", "2026-04-24T00:00:00Z",
        ],
        capture_output=True,
        text=True,
        cwd=str(paper_root),
    )
    assert result.returncode == 0, result.stderr

    with open(out_csv, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    expected = [
        "benchmark_id",
        "subject_id",
        "seed_id",
        "fewshot_k",
        "example_rank",
        "example_item_id",
        "example_source_split",
        "draw_timestamp",
        "draw_script_version",
    ]
    assert header == expected, (
        f"CSV header mismatch.\nexpected: {expected}\nactual:   {header}"
    )
    assert expected == MANIFEST_FIELDNAMES, (
        "MANIFEST_FIELDNAMES module constant has drifted from the spec."
    )
