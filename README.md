# LLM-HyperOpt

**Hyperparameter optimization with a large language model in the loop.**

Instead of random or Bayesian search, an LLM proposes each configuration — reading your
training code and reasoning about *why* previous trials did what they did. Within small
budgets this matches or beats Bayesian optimization ([Zhang et al., 2023](https://arxiv.org/abs/2312.04528)).

This repo ships two things:

- **Skills** (`.claude/skills/`) — the interactive tuner. You point an agent at a training
  command and it runs the search, explaining every choice.
- **A ledger core** (`llm_hyperopt/`) — a small, dependency-light Python package that
  records trials to disk so long, background searches are durable, resumable, and
  reproducible. It does no reasoning and calls no model; the reasoning lives in the skill.

The original research code is preserved under [`research/`](research/).

## How it works

The skill *is* the optimizer. It reads your code, proposes a search space, shows you the
plan, launches trials as background jobs, reads each validation metric, reasons over the
trajectory, and repeats until your budget is spent. The ledger keeps the state:

```
You ──▶ /tune ──▶ reads code, proposes search space, waits for your OK
                     │
                     ▼
        ┌───────────────────────────┐      records every trial
        │  propose config (reasoned) │◀───────────────┐
        │  launch training (bg job)  │                │
        │  read metric from log      │        llm_hyperopt ledger
        │  reason, pick next         │──────▶  .llm_hyperopt/<study>/study.json
        └───────────────────────────┘         (durable · resumable · inspectable)
```

## Quick start

```bash
pip install -e .
```

Then, in an agent (Claude Code / claude.ai), point the `tune` skill at a command that
prints the metric you care about:

> /tune — I run `python examples/sklearn_tabular/train.py --learning-rate 0.1 --n-estimators 200 --max-depth 3`.
> Maximize `val_acc` (printed to stdout). Tune learning-rate, n-estimators, max-depth. Budget 15.

The skill reads `train.py`, proposes ranges, shows you a plan, waits for your go-ahead,
then runs the search — narrating why it picks each configuration. See [`examples/`](examples/).

**Zero-contract:** you don't change your training code. It just needs to print the metric
on a line (e.g. `val_acc=0.9650`) and take hyperparameters as flags.

## Two audiences

- **Experts** — you have a training command and a validation metric. Use `tune`. It adapts
  to your existing code; nothing to rewrite.
- **Domain scientists** — you have a dataset and a goal but no training script. Use
  `tune-dataset`. It writes a transparent training script, explains the modeling choices,
  then hands off to `tune`. Pedagogical by default.

## The ledger core

Long trials run in the background across many agent turns, so trial state must be durable.
The `llm_hyperopt` package owns exactly that:

```python
from llm_hyperopt import Study

study = Study.create(name="cifar-vit", direction="maximize", metric="val_acc")
t = study.add_trial({"lr": 0.012, "wd": 1e-4, "bs": 64})  # recorded as running
# ... training job finishes ...
study.complete_trial(t.id, 0.83)
print(study.best().params, study.report())
```

Inspect any run from the terminal, no agent required:

```bash
llm-hyperopt report cifar-vit
llm-hyperopt best cifar-vit
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Research & citation

The method and experiments are in [`research/`](research/) and the paper. If you use this
work, please cite:

```bibtex
@article{zhang2023using,
  title={Using Large Language Models for Hyperparameter Optimization},
  author={Zhang, Michael R and Desai, Nishkrit and Bae, Juhan and Lorraine, Jonathan and Ba, Jimmy},
  journal={arXiv preprint arXiv:2312.04528},
  year={2023}
}
```

## License

MIT
