# ABOUTME: Example training script for the /tune skill — gradient boosting on a real
# ABOUTME: sklearn dataset, with tunable hyperparameters as CLI flags and a printed metric.

"""Train a gradient-boosted classifier on the breast-cancer dataset.

This is a deliberately simple, honest example of the zero-contract shape the `tune`
skill expects: hyperparameters are plain command-line flags, and the single number to
optimize is printed on its own line as `val_acc=<float>`. Nothing here knows about
LLMs or the tuner — that's the point.

Run one configuration by hand:

    python train.py --learning-rate 0.1 --n-estimators 200 --max-depth 3

Then tune it by pointing the /tune skill at that command, telling it to maximize
val_acc (parsed from stdout).
"""

import argparse

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_breast_cancer()
    x_train, x_val, y_train, y_val = train_test_split(
        data.data, data.target, test_size=0.25, random_state=args.seed
    )

    model = GradientBoostingClassifier(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        subsample=args.subsample,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)

    val_acc = model.score(x_val, y_val)
    # The tuner reads exactly this line.
    print(f"val_acc={val_acc:.4f}")


if __name__ == "__main__":
    main()
