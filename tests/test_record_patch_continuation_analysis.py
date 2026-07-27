from orcacolony.record_patch_continuation_analysis import (
    _build_parser,
    _conditioning_analysis,
    _optimizer_analysis,
    _simple_benchmarks,
)


def test_conditioning_analysis_uses_strict_json_scalar_types() -> None:
    examples = [
        {
            "id": "one",
            "record": {"a": True, "keep": "yes"},
            "target": '{"a":1,"keep":"yes"}',
        },
        {
            "id": "two",
            "record": {"drop": 2, "keep": None},
            "target": '{"keep":null}',
        },
    ]
    samples = [
        {
            "id": "one",
            "output": '{"a":true,"keep":"yes"}',
            "parsed_output": {"a": True, "keep": "yes"},
            "valid_json": True,
        },
        {
            "id": "two",
            "output": '{"keep":null,"keep":null}',
            "parsed_output": None,
            "valid_json": False,
        },
    ]

    analysis = _conditioning_analysis(samples, examples)

    assert analysis["valid_object_examples"] == 1
    assert analysis["expected_key_recall"] == {
        "matched": 2,
        "total": 3,
        "ratio": 2 / 3,
    }
    assert analysis["expected_key_value_recall"] == {
        "matched": 1,
        "total": 3,
        "ratio": 1 / 3,
    }
    assert analysis["output_key_value_pairs_copied_from_input"] == {
        "matched": 2,
        "total": 2,
        "ratio": 1.0,
    }


def test_simple_benchmark_compares_canonical_output_exactly() -> None:
    scores = _simple_benchmarks(
        [
            {
                "record": {"value": True},
                "target": '{"value":1}',
            },
            {
                "record": {"value": 1},
                "target": '{"value":1}',
            },
        ]
    )

    assert scores["copy_input_record"]["exact_matches"] == 1
    assert scores["deterministic_task_oracle"]["exact_matches"] == 2


def test_optimizer_analysis_requires_contiguous_steps() -> None:
    row = {
        "step": 129,
        "clipped": False,
        "training_mean_loss": 1.0,
        "gradient_global_norm_before_clipping": 0.5,
        "update_global_norm": 0.1,
        "relative_update_global_norm": 0.01,
    }
    analysis = _optimizer_analysis(
        [
            row,
            {
                **row,
                "step": 130,
                "clipped": True,
                "training_mean_loss": 0.9,
            },
        ]
    )

    assert analysis["steps"] == 2
    assert analysis["clipped_steps"] == 1
    assert analysis["first_step"] == 129
    assert analysis["last_step"] == 130


def test_analysis_cli_has_no_private_holdout_argument() -> None:
    destinations = {
        action.dest
        for action in _build_parser()._actions
    }

    assert "final_holdout" not in destinations
    assert "holdout_key" not in destinations
