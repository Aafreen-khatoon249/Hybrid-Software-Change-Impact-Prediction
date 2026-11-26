import pandas as pd
import numpy as np

from collections import defaultdict
from pathlib import Path


TRAIN_FILE = "dataset/temporal/train_commit_files.csv"
TEST_FILE = "dataset/temporal/test_commit_files.csv"

COCHANGE_FILE = "results/training_cochange_pairs.csv"
STATIC_FILE = "results/historical_static_dependencies.csv"

OUTPUT_FILE = Path(
    "results/evaluation/weight_experiments.csv"
)


def normalize(series):
    return (
        series.astype(str)
        .str.replace("\\", "/", regex=False)
    )


def run_experiment():

    print("Starting Hybrid Weight Experiments...")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    train = train[
        train["file_name"].str.endswith(".py", na=False)
    ].copy()

    test = test[
        test["file_name"].str.endswith(".py", na=False)
    ].copy()

    train["file_name"] = normalize(train["file_name"])
    test["file_name"] = normalize(test["file_name"])

    training_files = set(train["file_name"].unique())

    # Load co-change patterns
    cochange_data = pd.read_csv(COCHANGE_FILE)

    cochange = defaultdict(int)

    for _, row in cochange_data.iterrows():

        a = row["file_1"].replace("\\", "/")
        b = row["file_2"].replace("\\", "/")

        cochange[(a, b)] = int(row["cochange_count"])
        cochange[(b, a)] = int(row["cochange_count"])

    # Load static dependencies
    static_data = pd.read_csv(STATIC_FILE)

    graph = defaultdict(set)

    for _, row in static_data.iterrows():

        a = row["source_file"].replace("\\", "/")
        b = row["target_file"].replace("\\", "/")

        graph[a].add(b)
        graph[b].add(a)

    # Experiment weights
    weights = [0.25, 0.50, 0.75]

    results = []

    for commit_hash, group in test.groupby("commit_hash"):

        changed = sorted(set(group["file_name"]))

        if len(changed) < 2:
            continue

        for source in changed:

            actual = (set(changed) - {source}) & training_files

            if not actual:
                continue

            candidates = training_files - {source}

            static_scores = {
                f: int(f in graph[source])
                for f in candidates
            }

            raw_scores = {
                f: cochange.get((source, f), 0)
                for f in candidates
            }

            maximum = max(raw_scores.values(), default=0)

            cochange_scores = {
                f: (
                    raw_scores[f] / maximum
                    if maximum > 0 else 0
                )
                for f in candidates
            }

            for static_weight in weights:

                cochange_weight = 1 - static_weight

                scores = {
                    f: (
                        static_weight * static_scores[f]
                        + cochange_weight * cochange_scores[f]
                    )
                    for f in candidates
                }

                ranking = sorted(
                    candidates,
                    key=lambda f: (scores[f], f),
                    reverse=True
                )

                for k in [5, 10]:

                    recommended = set(ranking[:k])

                    hits = len(recommended & actual)

                    precision = hits / k
                    recall = hits / len(actual)

                    f1 = (
                        2 * precision * recall
                        / (precision + recall)
                        if precision + recall else 0
                    )

                    results.append({
                        "static_weight": static_weight,
                        "cochange_weight": cochange_weight,
                        "k": k,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    })

    df = pd.DataFrame(results)

    summary = (
        df.groupby(
            ["static_weight", "cochange_weight", "k"]
        )[
            ["precision", "recall", "f1_score"]
        ]
        .mean()
        .reset_index()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(OUTPUT_FILE, index=False)

    print("\n========== WEIGHT EXPERIMENT RESULTS ==========")

    print(summary.to_string(index=False))

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_experiment()