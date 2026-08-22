from datetime import UTC, datetime
from pathlib import Path

from ire_assn1.experiments.manifests import RunManifest, StageRecord, artifact
from ire_assn1.experiments.stages import _reusable_stage, run_stage


def _manifest(stage: StageRecord) -> RunManifest:
    return RunManifest(
        run_id="test-run",
        created_at=datetime.now(UTC),
        git_sha="test-sha",
        git_dirty=False,
        reportable=True,
        source_kind="git",
        source_tree_sha256="tree-sha",
        config_sha256="config-sha",
        lock_sha256="lock-sha",
        resolved_config={},
        python_version="3.12",
        platform="test",
        cpu_count=1,
        memory_bytes=1,
        device={},
        stages=[stage],
    )


def test_completed_stage_is_reusable_only_when_artifacts_match(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text("input", encoding="utf-8")
    output_path.write_text("output", encoding="utf-8")
    timestamp = datetime.now(UTC)
    manifest = _manifest(
        StageRecord(
            name="stage",
            state="complete",
            started_at=timestamp,
            inputs=[artifact(input_path)],
            outputs=[artifact(output_path)],
        )
    )

    called = False

    def operation() -> None:
        nonlocal called
        called = True

    run_stage(manifest, "stage", operation, (input_path,), (output_path,), resume=True)
    assert not called
    assert _reusable_stage(manifest, "stage", (input_path,), (output_path,))

    output_path.write_text("changed", encoding="utf-8")
    assert not _reusable_stage(manifest, "stage", (input_path,), (output_path,))
