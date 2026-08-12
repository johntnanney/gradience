"""Regression tests for legacy Hugging Face dataset id normalization.

The Hub stopped accepting bare canonical dataset ids ("glue", "gsm8k", ...) and
now requires "namespace/name". Configs and recorded provenance still carry the
legacy short ids, so normalization happens at the load boundary.
"""

import pytest

from gradience.bench._util import canonical_dataset_id


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("glue", "nyu-mll/glue"),
        ("gsm8k", "openai/gsm8k"),
        ("sst2", "stanfordnlp/sst2"),
        ("GLUE", "nyu-mll/glue"),
    ],
)
def test_legacy_ids_are_namespaced(legacy, expected):
    assert canonical_dataset_id(legacy) == expected


@pytest.mark.parametrize(
    "already_namespaced",
    ["nyu-mll/glue", "openai/gsm8k", "some-org/private-dataset"],
)
def test_namespaced_ids_pass_through(already_namespaced):
    assert canonical_dataset_id(already_namespaced) == already_namespaced


@pytest.mark.parametrize("unknown", ["", "not_a_known_dataset"])
def test_unknown_ids_pass_through(unknown):
    assert canonical_dataset_id(unknown) == unknown


def test_every_mapped_target_is_namespaced():
    from gradience.bench._util import LEGACY_DATASET_IDS

    for legacy, target in LEGACY_DATASET_IDS.items():
        assert "/" not in legacy, f"legacy key {legacy!r} should be a bare id"
        assert "/" in target, f"target {target!r} must be namespace/name"
