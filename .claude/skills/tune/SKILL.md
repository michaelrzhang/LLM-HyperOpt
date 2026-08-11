---
name: tune
description: >-
  Tune the hyperparameters of any training script or ML repo with an LLM in the
  loop — you propose configurations, run them, read the validation metric, reason
  over the trajectory, and repeat until a budget is spent. Use whenever the user
  wants to optimize/tune/search hyperparameters, find a good learning rate or
  config, or improve a model's validation metric, given a command they can run.
  Zero-contract: the user changes nothing in their code.
---

# Tune: LLM-in-the-loop hyperparameter optimization

You are the optimizer. Instead of random or Bayesian search, *you* propose each
configuration using what you can read from the code and what you've learned from
previous trials. This method — prompting an LLM to iteratively suggest configs
refined on observed performance — is the one from *"Using Large Language Models
for Hyperparameter Optimization"* (Zhang et al., 2023). Within small budgets it
matches or beats Bayesian optimization; your edge is reading the code and
reasoning about *why* a config did what it did.

The `llm_hyperopt` package is a **ledger only** — it records trials and survives
across turns. It does no reasoning and calls no model. The reasoning is yours.

## The five phases

1. **Discover** — figure out the run command, the tunable knobs, and where the metric appears.
2. **Plan & confirm** — show the search space, metric, budget, and round-1 strategy. Wait for a yes.
3. **Launch** — start trials as background jobs; record each as `running` in the ledger.
4. **Observe & iterate** — when a job finishes, read its metric, reason, launch the next.
5. **Report** — when the budget is spent or progress stalls, summarize and hand back the best config.

Never skip phase 2. Launching training jobs costs the user time and money.

---

## Phase 1 — Discover (zero-contract)

The user changes nothing in their code. You need three things:

- **Run command** — how they launch one training run, e.g. `python train.py --lr 0.01 --bs 64`.
- **Tunable knobs** — which flags/config values to search, and plausible ranges. Read the
  training script and any config files to infer these. Prefer log-scale for learning
  rate, weight decay, and anything spanning orders of magnitude.
- **The metric** — the number to optimize, where it appears (a stdout line like
  `val_acc=0.83`, a JSON/CSV log, a TensorBoard/W&B file), and the **direction**
  (maximize accuracy/AUC, minimize loss/error).

Ask the user for anything you cannot confidently infer — especially the metric
location and direction. Do not guess the direction.

If knobs are set via a **config file** rather than CLI flags, plan to edit a copy of
that file per trial instead of passing flags. Never mutate the user's original config
or code in place; write per-trial copies under the study directory.

---

## Phase 2 — Plan & confirm (required gate)

Summarize your plan in plain language and stop for confirmation. Include:

- The search space, with a one-line rationale per knob and why each range/scale.
- The metric and direction, and exactly how you'll read it.
- The budget (number of trials) and how many run concurrently (default **3**).
- What round 1 will do (usually: spread configs across the ranges to locate the
  good region before exploiting it).

Example:

```
Here's my plan for tuning train.py:
  • lr   — 1e-5 … 1e-1, log scale (dominates convergence; wide because ViTs are lr-sensitive)
  • wd   — 0 … 1e-3 (mild regularization; small model)
  • bs   — {32, 64, 128} (interacts with lr)
  Metric : val_acc, maximize, parsed from stdout "val_acc=<float>"
  Budget : 20 trials, up to 3 running at once
  Round 1: 3 configs spread across the lr range to find the basin
Proceed?
```

Only after the user confirms, create the study and start launching.

---

## Phase 3 — Launch trials

Create (or resume) the study, then launch each trial as a **background job** so long
runs don't block the turn. Record it as `running` before it starts.

Pick a short study name (e.g. from the script or task). Create the study once:

```bash
python -c "
from llm_hyperopt import Study
Study.create(name='STUDY', direction='maximize', metric='val_acc')
"
```

For each trial you decide to run:

```bash
# 1) register the config; capture the trial id
TID=$(python -c "
from llm_hyperopt import Study
s = Study.load('STUDY')
t = s.add_trial({'lr': 0.012, 'wd': 1e-4, 'bs': 64},
                log_path='.llm_hyperopt/STUDY/logs/trial-PLACEHOLDER.log')
print(t.id)
")
```

Then launch the user's real command with those values substituted, redirecting all
output to that trial's log, as a **background** Bash call (`run_in_background: true`).
Use the real trial id in the log filename. The harness re-invokes you when the job
exits — that is your signal to move to phase 4 for that trial.

Keep up to **3** trials running at once (or whatever the user set). Don't launch the
whole budget at once — each new config should benefit from what earlier ones taught you.

---

## Phase 4 — Observe & iterate

When a background job finishes, for that trial:

1. **Read the metric** from its log. Parse robustly — grep the agreed pattern, or read
   the JSON/CSV/event file. If several values appear (per-epoch), take the one the user
   cares about (usually best or final validation).
2. **Record the outcome:**

   ```bash
   python -c "
   from llm_hyperopt import Study
   s = Study.load('STUDY')
   s.complete_trial(TID, 0.83)          # or: s.fail_trial(TID, 'diverged: NaN loss')
   print(s.report())
   "
   ```

3. **Reason, out loud and briefly**, over the whole trajectory, then propose the next
   config and launch it (phase 3) to keep the pipeline full. Tell the user *why* — this
   transparency is the point of the tool.

### How to reason about the next config

- **Early**: spread across the ranges, especially log-scale ones, to find where the
  metric is good at all. Don't over-exploit before you've located the basin.
- **Then exploit**: sample near the best region, nudging one or two knobs at a time so
  you can attribute the change.
- **Watch interactions**: lr×batch-size and lr×weight-decay are the classic ones.
- **Learn from failures**: divergence/NaN usually means lr (or lr/warmup) too high —
  lower the ceiling and record that with `fail_trial` so it's on the record.
- **Respect the budget**: near the end, stop exploring and confirm the best region.
- **Stop early** if the metric has clearly plateaued and remaining budget won't help;
  say so rather than burning trials.

Never fabricate a metric. If a log is unreadable or the run crashed, `fail_trial` it
with the reason and adjust — do not invent a number.

---

## Phase 5 — Report

When the budget is spent (or you've stopped early), print `Study.report()`, then give:

- The best configuration and its metric.
- The **exact command** to reproduce that run.
- A short note on what mattered (e.g. "lr was decisive; wd barely moved the metric").

---

## Resuming

If a study already exists (e.g. a new session, or the user comes back later), don't
start over:

```bash
python -c "from llm_hyperopt import Study; print(Study.load('STUDY').report())"
```

Check which trials are still `running`, verify whether their jobs are actually alive,
complete or fail them from their logs, and continue the loop from there.

## Notes

- Trial state lives under `.llm_hyperopt/<study>/`. Suggest adding it to `.gitignore`.
- Inspect any study without an agent: `llm-hyperopt report <study>` / `llm-hyperopt best <study>`.
- Model-agnostic: the reasoning is done by whatever model is running you.
