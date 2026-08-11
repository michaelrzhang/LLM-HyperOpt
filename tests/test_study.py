# ABOUTME: Tests for the Study ledger — the persistent record of trials that
# ABOUTME: survives across agent turns and makes background optimization resumable.

import json
import os

import pytest

from llm_hyperopt import Study, Trial


def test_create_study_persists_immediately(tmp_path):
    study = Study.create(
        name="demo", direction="maximize", metric="val_acc", storage_root=tmp_path
    )
    assert study.name == "demo"
    assert study.direction == "maximize"
    assert study.metric == "val_acc"
    # Creating a study writes it to disk so a later agent turn can load it.
    assert (tmp_path / "demo" / "study.json").exists()


def test_direction_must_be_valid(tmp_path):
    with pytest.raises(ValueError):
        Study.create(name="x", direction="highest", metric="m", storage_root=tmp_path)


def test_add_trial_starts_running_with_sequential_ids(tmp_path):
    study = Study.create(
        name="demo", direction="maximize", metric="val_acc", storage_root=tmp_path
    )
    t1 = study.add_trial({"lr": 0.01, "bs": 64}, log_path="runs/1.log")
    t2 = study.add_trial({"lr": 0.03, "bs": 32})

    assert t1.id == 1
    assert t2.id == 2
    assert t1.status == "running"
    assert t1.metric is None
    assert t1.log_path == "runs/1.log"
    assert study.running() == [t1, t2]


def test_complete_trial_records_metric(tmp_path):
    study = Study.create(
        name="demo", direction="maximize", metric="val_acc", storage_root=tmp_path
    )
    t = study.add_trial({"lr": 0.01})
    study.complete_trial(t.id, 0.83)

    done = study.completed()
    assert len(done) == 1
    assert done[0].metric == 0.83
    assert done[0].status == "completed"
    assert done[0].completed_at is not None
    assert study.running() == []


def test_fail_trial_records_error(tmp_path):
    study = Study.create(
        name="demo", direction="maximize", metric="val_acc", storage_root=tmp_path
    )
    t = study.add_trial({"lr": 10.0})
    study.fail_trial(t.id, "diverged: loss became NaN")

    assert study.running() == []
    assert study.completed() == []
    failed = [x for x in study.trajectory() if x.status == "failed"]
    assert len(failed) == 1
    assert "NaN" in failed[0].error


def test_best_respects_maximize(tmp_path):
    study = Study.create(
        name="demo", direction="maximize", metric="val_acc", storage_root=tmp_path
    )
    a = study.add_trial({"lr": 0.01})
    b = study.add_trial({"lr": 0.03})
    study.complete_trial(a.id, 0.70)
    study.complete_trial(b.id, 0.85)
    assert study.best().id == b.id
    assert study.best().metric == 0.85


def test_best_respects_minimize(tmp_path):
    study = Study.create(
        name="demo", direction="minimize", metric="val_loss", storage_root=tmp_path
    )
    a = study.add_trial({"lr": 0.01})
    b = study.add_trial({"lr": 0.03})
    study.complete_trial(a.id, 0.40)
    study.complete_trial(b.id, 0.25)
    assert study.best().id == b.id


def test_best_ignores_running_and_failed(tmp_path):
    study = Study.create(
        name="demo", direction="maximize", metric="val_acc", storage_root=tmp_path
    )
    running = study.add_trial({"lr": 0.01})  # never completed
    failed = study.add_trial({"lr": 9.0})
    study.fail_trial(failed.id, "boom")
    assert study.best() is None
    done = study.add_trial({"lr": 0.02})
    study.complete_trial(done.id, 0.5)
    assert study.best().id == done.id
    # sanity: the running trial is still there and untouched
    assert running.status == "running"


def test_load_resumes_from_disk(tmp_path):
    study = Study.create(
        name="resume-me", direction="maximize", metric="val_acc", storage_root=tmp_path
    )
    a = study.add_trial({"lr": 0.01})
    study.complete_trial(a.id, 0.83)
    b = study.add_trial({"lr": 0.05})  # left running across the "turn boundary"

    reloaded = Study.load("resume-me", storage_root=tmp_path)
    assert reloaded.metric == "val_acc"
    assert reloaded.best().metric == 0.83
    assert len(reloaded.running()) == 1
    assert reloaded.running()[0].params == {"lr": 0.05}
    # ids keep incrementing after a resume rather than colliding
    c = reloaded.add_trial({"lr": 0.07})
    assert c.id == 3


def test_load_missing_study_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Study.load("nope", storage_root=tmp_path)


def test_saved_json_is_human_readable(tmp_path):
    study = Study.create(
        name="demo", direction="minimize", metric="val_loss", storage_root=tmp_path
    )
    t = study.add_trial({"lr": 0.01, "optimizer": "adamw"})
    study.complete_trial(t.id, 0.31)

    data = json.loads((tmp_path / "demo" / "study.json").read_text())
    assert data["direction"] == "minimize"
    assert data["metric"] == "val_loss"
    assert data["trials"][0]["params"]["optimizer"] == "adamw"
    assert data["trials"][0]["metric"] == 0.31


def test_report_mentions_best_and_counts(tmp_path):
    study = Study.create(
        name="demo", direction="maximize", metric="val_acc", storage_root=tmp_path
    )
    a = study.add_trial({"lr": 0.01})
    study.complete_trial(a.id, 0.80)
    study.add_trial({"lr": 0.02})  # still running
    report = study.report()
    assert "val_acc" in report
    assert "0.80" in report or "0.8000" in report
    assert "running" in report.lower()
