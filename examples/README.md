# Examples

Every example follows the same zero-contract shape the `tune` skill expects: a normal
training command whose hyperparameters are CLI flags, printing the metric to optimize on
one line. You change nothing in the training code to tune it.

## Runnable now: `sklearn_tabular/`

Gradient boosting on the real breast-cancer dataset — a few seconds per trial, no GPU.

```bash
pip install scikit-learn
python examples/sklearn_tabular/train.py --learning-rate 0.1 --n-estimators 200 --max-depth 3
# -> val_acc=0.9650
```

Tune it by pointing the skill at that command:

> /tune — I run `python examples/sklearn_tabular/train.py --learning-rate 0.1 --n-estimators 200 --max-depth 3`.
> Maximize `val_acc` (printed to stdout). Tune learning-rate, n-estimators, max-depth, subsample. Budget 15.

The skill will read `train.py`, propose ranges (log-scale learning rate, etc.), show you
the plan, then run the search and explain each pick.

## Heavier scenarios (same shape, real workloads)

These need real compute; the pattern is identical — a run command and a printed metric.

- **Karpathy's nanoGPT** — tune learning rate, warmup, weight decay, and dropout against
  `train.py`'s printed validation loss (minimize). Trials are minutes–hours on a GPU, so
  the skill launches them in the background and reasons over `val_loss` as each returns.
- **Finance** — tune a returns-forecasting model; optimize validation Sharpe or IC
  (maximize) with a strict walk-forward split to avoid look-ahead leakage.
- **Science** — tune a genomics or materials-property model; optimize validation AUC or
  MAE. The pedagogical `tune-dataset` skill can write the training script first if the
  researcher only has a dataset.

## The contract, in one line

Your training command prints the metric like `val_acc=0.9650` (or `val_loss=0.42`), and
you tell the skill which flags to search and whether to maximize or minimize. That's it.
