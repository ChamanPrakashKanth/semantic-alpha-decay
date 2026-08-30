import json

from gru_linear_small_model.curriculum import CurriculumStage, curriculum_stages
from gru_linear_small_model.curriculum_compare import run_comparison


def test_curriculum_stages_progress_monotonically_in_difficulty():
    stages = curriculum_stages("pilot")
    assert [stage.stage for stage in stages] == list(range(7))
    assert stages[0].chain_lengths == [1]
    assert stages[0].edge_order == "ordered"
    assert stages[-1].destination_coverage == 24
    assert stages[-1].edge_order == "shuffled"


def test_micro_comparison_uses_shared_schedule_and_preserves_test(tmp_path):
    stages = [
        CurriculumStage(0, "micro", [1], "ordered", [0], 0, 0.0, 1),
    ]
    result = run_comparison(
        tmp_path,
        seed=3,
        embedding_dim=4,
        hidden_dim=4,
        batch_size=2,
        validation_examples=4,
        final_probe_examples=4,
        validate_every=1,
        required_consecutive_passes=1,
        stages_override=stages,
    )
    assert result["all_stages_passed"]
    assert result["shared_batch_training"] is True
    assert result["stage_controller"] == "gru_validation_only"
    assert result["reserved_test_was_evaluated"] is False
    assert len(result["stage_reports"]) == 1
    stored = json.loads((tmp_path / "comparison_report.json").read_text())
    assert stored["reserved_untouched_test_seed"] == 990003
    assert (tmp_path / "diagnosis.md").exists()
    assert (tmp_path / "gru_hybrid_curriculum.pt").exists()
    assert (tmp_path / "shared_schedule.json").exists()
    assert (tmp_path / "stage_results.csv").exists()
