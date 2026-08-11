# ABOUTME: Tests for the headless `llm-hyperopt` CLI used to inspect studies without an agent.

import json

from llm_hyperopt import Study
from llm_hyperopt.cli import main


def _seed_study(tmp_path):
    study = Study.create(
        name="demo", direction="maximize", metric="val_acc", storage_root=tmp_path
    )
    a = study.add_trial({"lr": 0.01})
    study.complete_trial(a.id, 0.80)
    b = study.add_trial({"lr": 0.03})
    study.complete_trial(b.id, 0.91)
    study.add_trial({"lr": 0.05})  # still running
    return study


def test_report_command(tmp_path, capsys):
    _seed_study(tmp_path)
    rc = main(["report", "demo", "--storage-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "val_acc" in out
    assert "running" in out.lower()


def test_best_command_emits_json(tmp_path, capsys):
    _seed_study(tmp_path)
    rc = main(["best", "demo", "--storage-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["metric"] == 0.91
    assert payload["params"] == {"lr": 0.03}


def test_list_command(tmp_path, capsys):
    _seed_study(tmp_path)
    rc = main(["list", "--storage-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "demo" in out


def test_report_missing_study_returns_nonzero(tmp_path, capsys):
    rc = main(["report", "ghost", "--storage-root", str(tmp_path)])
    assert rc != 0
