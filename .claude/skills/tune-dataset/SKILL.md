---
name: tune-dataset
description: >-
  Go from a raw dataset to a tuned model for someone who knows their domain but not
  ML engineering. Use when the user has data (CSV, images, a table) and a goal
  ("predict X", "classify Y") but no training script yet. You write a clear,
  transparent training script, explain the modeling choices, then hand off to the
  `tune` skill to optimize it. Pedagogical by default.
---

# Tune-dataset: from data to a tuned model, explained

This is the non-expert on-ramp. The user is a scientist or analyst who understands
their data and what they want to predict, but hasn't written the training code. Your
job is to get them a good model *and* leave them understanding how it works. It
reduces to the expert `tune` skill once a training script exists.

Teach as you go. Prefer the simplest model that fits the problem, and say why.

## Steps

1. **Understand the problem.** What are the inputs and the target? Is it
   classification or regression? How much data? Ask; don't assume. Look at the file.

2. **Explain the plan before coding.** In a few sentences: the model you'll start with
   and why (default to a strong, simple baseline — gradient-boosted trees for tabular,
   a small CNN or a fine-tuned backbone for images), how you'll split train/validation,
   and which metric matches their goal (and its direction). Confirm the metric with them
   — it encodes what "good" means.

3. **Write a transparent training script.** Create a real, runnable `train.py` that:
   - loads their dataset and makes a proper train/validation split (no leakage),
   - exposes the hyperparameters you'll tune as **command-line flags**,
   - trains and prints the validation metric in a single clear line, e.g.
     `print(f"val_auc={score:.4f}")`.
   Comment it so they can read it. Use real libraries and real data — never mock or
   fabricate results. Run it once end-to-end to confirm it works and produces a metric.

4. **Explain what each hyperparameter does** in plain language before tuning — learning
   rate, regularization, model size — so the tuning that follows isn't a black box.

5. **Hand off to `tune`.** With a working script that prints a metric, invoke the `tune`
   skill using that run command, the flags as the search space, and the metric line.
   Keep narrating the reasoning as trials come back so they learn from the search itself.

6. **Close the loop.** Report the best model, how to reproduce it, and — briefly — what
   the search revealed about their problem (which knobs mattered). Suggest a sensible
   next experiment.

## Guardrails

- Small dataset → prefer simpler models and cross-validation; warn about overfitting.
- Be honest about limits: if the data is too small or the signal weak, say so.
- Everything real: real data, real training, real metrics. No shortcuts, no mocks.
