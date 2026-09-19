import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parent / "scripts" / "parley_engine.py"
spec = importlib.util.spec_from_file_location("parley_engine", MODULE_PATH)
parley_engine = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(parley_engine)


def test_build_steps_preserves_deterministic_order():
    request = {
        "mission_create": {"title": "x"},
        "plan_decompose": {"mission_id": "m1"},
        "placement_scores": [
            {"mission_id": "m1", "node_id": "a"},
            {"mission_id": "m1", "node_id": "b"},
        ],
    }

    steps = parley_engine.build_steps(request)

    assert [step.tool for step in steps] == [
        "hermes_mission_create",
        "hermes_plan_decompose",
        "hermes_placement_score",
        "hermes_placement_score",
    ]


def test_unknown_key_is_fail_closed():
    with pytest.raises(ValueError, match="unknown request keys"):
        parley_engine.build_steps({"a2a_dispatch": {"agent": "server"}})


def test_repeated_step_requires_list():
    with pytest.raises(ValueError, match="placement_scores must be a list"):
        parley_engine.build_steps({"placement_scores": {"node_id": "x"}})


def test_empty_request_is_rejected():
    with pytest.raises(ValueError, match="no Mission Control steps"):
        parley_engine.build_steps({})


def test_prepare_arguments_forces_dry_run_by_default():
    step = parley_engine.Step("plan_create", "hermes_plan_create", {"dry_run": False, "mission_id": "m1"})
    schema = {"properties": {"dry_run": {"type": "boolean"}, "mission_id": {"type": "string"}}}

    args = parley_engine.prepare_arguments(step, schema, apply=False)

    assert args["dry_run"] is True


def test_prepare_arguments_apply_only_changes_supported_dry_run():
    step = parley_engine.Step("plan_create", "hermes_plan_create", {"mission_id": "m1"})

    with_dry_run = parley_engine.prepare_arguments(
        step,
        {"properties": {"dry_run": {"type": "boolean"}}},
        apply=True,
    )
    without_dry_run = parley_engine.prepare_arguments(
        step,
        {"properties": {"mission_id": {"type": "string"}}},
        apply=True,
    )

    assert with_dry_run["dry_run"] is False
    assert "dry_run" not in without_dry_run


def test_allowlist_contains_no_dispatch_or_approval_tools():
    lowered = " ".join(sorted(parley_engine.ALLOWED_TOOLS)).lower()
    assert "dispatch" not in lowered
    assert "approve" not in lowered
    assert "runner" not in lowered
    assert "shell" not in lowered
