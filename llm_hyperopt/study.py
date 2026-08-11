# ABOUTME: The Study ledger — a persistent, human-readable record of optimization
# ABOUTME: trials that survives across agent turns and makes background runs resumable.

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# A trial moves through exactly these states over its life.
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"

_VALID_DIRECTIONS = ("maximize", "minimize")


@dataclass
class Trial:
    """A single hyperparameter evaluation.

    A trial is created in the RUNNING state when its job is launched, then later
    moved to COMPLETED (with a metric) or FAILED (with an error) once the job ends.
    """

    id: int
    params: Dict[str, Any]
    status: str = RUNNING
    metric: Optional[float] = None
    log_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trial":
        return cls(**data)


class Study:
    """The ledger for one optimization run.

    The Study owns no reasoning and calls no LLM. It records what was tried and
    what happened, persisting to disk on every change so that a background trial
    launched in one agent turn can be completed and reasoned about in a later one.
    """

    def __init__(
        self,
        name: str,
        direction: str,
        metric: str,
        storage_root: Union[str, Path] = ".llm_hyperopt",
        trials: Optional[List[Trial]] = None,
        autosave: bool = True,
    ) -> None:
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {_VALID_DIRECTIONS}, got {direction!r}"
            )
        self.name = name
        self.direction = direction
        self.metric = metric
        self.storage_root = Path(storage_root)
        self.trials: List[Trial] = trials if trials is not None else []
        self.autosave = autosave

    # -- construction / persistence ------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        direction: str,
        metric: str,
        storage_root: Union[str, Path] = ".llm_hyperopt",
        autosave: bool = True,
    ) -> "Study":
        """Start a fresh study and write it to disk so it can be loaded later."""
        study = cls(
            name=name,
            direction=direction,
            metric=metric,
            storage_root=storage_root,
            autosave=autosave,
        )
        study.save()
        return study

    @classmethod
    def load(
        cls, name: str, storage_root: Union[str, Path] = ".llm_hyperopt"
    ) -> "Study":
        """Resume an existing study from disk."""
        path = Path(storage_root) / name / "study.json"
        if not path.exists():
            raise FileNotFoundError(f"No study found at {path}")
        data = json.loads(path.read_text())
        trials = [Trial.from_dict(t) for t in data.get("trials", [])]
        return cls(
            name=data["name"],
            direction=data["direction"],
            metric=data["metric"],
            storage_root=storage_root,
            trials=trials,
        )

    @property
    def path(self) -> Path:
        return self.storage_root / self.name / "study.json"

    def save(self) -> None:
        """Atomically persist the study to ``<storage_root>/<name>/study.json``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "direction": self.direction,
            "metric": self.metric,
            "trials": [t.to_dict() for t in self.trials],
        }
        # Write to a temp file then replace so an interrupted write never leaves
        # a half-written ledger behind.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)

    def _maybe_save(self) -> None:
        if self.autosave:
            self.save()

    # -- mutation ------------------------------------------------------------

    def add_trial(
        self,
        params: Dict[str, Any],
        log_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Trial:
        """Record a newly launched trial in the RUNNING state."""
        trial = Trial(
            id=self._next_id(),
            params=params,
            status=RUNNING,
            log_path=log_path,
            metadata=metadata or {},
        )
        self.trials.append(trial)
        self._maybe_save()
        return trial

    def complete_trial(self, trial_id: int, metric: float) -> Trial:
        """Mark a trial finished and record its validation metric."""
        trial = self.get(trial_id)
        trial.status = COMPLETED
        trial.metric = metric
        trial.completed_at = time.time()
        self._maybe_save()
        return trial

    def fail_trial(self, trial_id: int, error: str) -> Trial:
        """Mark a trial failed and record why (e.g. divergence, OOM, crash)."""
        trial = self.get(trial_id)
        trial.status = FAILED
        trial.error = error
        trial.completed_at = time.time()
        self._maybe_save()
        return trial

    # -- queries -------------------------------------------------------------

    def get(self, trial_id: int) -> Trial:
        for trial in self.trials:
            if trial.id == trial_id:
                return trial
        raise KeyError(f"No trial with id {trial_id} in study {self.name!r}")

    def trajectory(self) -> List[Trial]:
        """All trials in the order they were launched."""
        return list(self.trials)

    def running(self) -> List[Trial]:
        return [t for t in self.trials if t.status == RUNNING]

    def completed(self) -> List[Trial]:
        return [t for t in self.trials if t.status == COMPLETED]

    def best(self) -> Optional[Trial]:
        """The best completed trial, respecting the optimization direction."""
        done = [t for t in self.completed() if t.metric is not None]
        if not done:
            return None
        if self.direction == "maximize":
            return max(done, key=lambda t: t.metric)
        return min(done, key=lambda t: t.metric)

    def report(self) -> str:
        """A human-readable summary of progress so far."""
        n_run = len(self.running())
        n_done = len(self.completed())
        n_fail = len([t for t in self.trials if t.status == FAILED])
        lines = [
            f"Study '{self.name}' — {self.direction} {self.metric}",
            f"  {n_done} completed, {n_run} running, {n_fail} failed",
        ]
        best = self.best()
        if best is not None:
            lines.append(f"  best: {self.metric}={best.metric:.4f} @ trial {best.id} {best.params}")
        else:
            lines.append("  best: (no completed trials yet)")
        lines.append("")
        lines.append("Trajectory:")
        for t in self.trajectory():
            if t.status == COMPLETED:
                detail = f"{self.metric}={t.metric:.4f}"
            elif t.status == FAILED:
                detail = f"failed: {t.error}"
            else:
                detail = "running"
            lines.append(f"  trial {t.id}: {detail}  {t.params}")
        return "\n".join(lines)

    # -- internals -----------------------------------------------------------

    def _next_id(self) -> int:
        return len(self.trials) + 1

    def __repr__(self) -> str:
        return (
            f"Study(name={self.name!r}, direction={self.direction!r}, "
            f"metric={self.metric!r}, trials={len(self.trials)})"
        )
