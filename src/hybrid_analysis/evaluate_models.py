import pandas as pd
import numpy as np

from collections import defaultdict
from itertools import combinations
from pathlib import Path


# Input datasets
TRAIN_FILE = "dataset/temporal/train_commit_files.csv"
TEST_FILE = "dataset/temporal/test_commit_files.csv"

COCHANGE_FILE = "results/training_cochange_pairs.csv"
STATIC_FILE = "results/historical_static_dependencies.csv"

OUTPUT_DIR = Path("results/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_paths(series):
    return (
        series.astype(str)
        .str.replace("\\", "/", regex=False)
    )


def load_static_graph():

    static = pd.read_csv(STATIC_FILE)

    static["source_file"] = normalize_paths(
        static["source_file"]
    )

    static["target_file"] = normalize_paths(
        static["target_file"]
    )

    graph = defaultdict(set)

    # Treat dependency relationships as undirected
    # for this initial recommendation baseline.
    for _, row in static.iterrows():

        source = row["source_file"]
        target = row["target_file"]

        graph[source].add(target)
        graph[target].add(source)

    return graph


def load_cochange_patterns():

    data = pd.read_csv(COCHANGE_FILE)

    data["file_1"] = normalize_paths(data["file_1"])
    data["file_2"] = normalize_paths(data["file_2"])

    patterns = defaultdict(int)

    for _, row in data.iterrows():

        file_1 = row["file_1"]
        file_2 = row["file_2"]

        count = int(row["cochange_count"])

        patterns[(file_1, file_2)] = count
        patterns[(file_2, file_1)] = count

    return patterns


def calculate_metrics(recommendations, actual, k):

    recommended = set(recommendations[:k])
    actual = set(actual)

    if not actual:
        return None

    hits = len(recommended & actual)

    precision = hits / k
    recall = hits / len(actual)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return precision, recall, f1


def evaluate():

    print("Loading training and testing data...")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    # Keep Python files only
    train = train[
        train["file_name"].str.endswith(".py", na=False)
    ].copy()

    test = test[
        test["file_name"].str.endswith(".py", na=False)
    ].copy()

    train["file_name"] = normalize_paths(train["file_name"])
    test["file_name"] = normalize_paths(test["file_name"])

    # Candidate files must have appeared in training
    training_files = set(train["file_name"].unique())

    # Load learned relationships
    cochange = load_cochange_patterns()
    static_graph = load_static_graph()

    # Test commits
    test_groups = test.groupby("commit_hash")

    # Store metric values
    metric_values = {
        "Static-only": {5: [], 10: []},
        "Co-change-only": {5: [], 10: []},
        "Hybrid": {5: [], 10: []}
    }

    evaluated_queries = 0
    skipped_queries = 0
    total_queries = 0

    # Evaluate each changed file in each test commit
    for commit_hash, group in test_groups:

        changed_files = sorted(
            set(group["file_name"])
        )

        # Need at least two files to create
        # a change-impact prediction example.
        if len(changed_files) < 2:
            continue

        for source_file in changed_files:

            total_queries += 1

            # Other files changed in the same commit
            actual_all = set(changed_files) - {source_file}

            # Restrict evaluation to files known in training.
            # Unseen targets cannot be recommended by these baselines.
            actual = actual_all & training_files

            if not actual:
                skipped_queries += 1
                continue

            candidates = training_files - {source_file}

            # Score candidates
            static_scores = {}
            cochange_scores = {}

            for candidate in candidates:

                static_scores[candidate] = (
                    1 if candidate in static_graph[source_file]
                    else 0
                )

                cochange_scores[candidate] = cochange.get(
                    (source_file, candidate),
                    0
                )

            # Normalize co-change scores per query
            max_score = max(cochange_scores.values(), default=0)

            if max_score > 0:
                normalized_cochange = {
                    file: score / max_score
                    for file, score in cochange_scores.items()
                }
            else:
                normalized_cochange = {
                    file: 0
                    for file in candidates
                }

            # Hybrid score: equal weight for both sources
            hybrid_scores = {
                file: (
                    0.5 * static_scores[file]
                    + 0.5 * normalized_cochange[file]
                )
                for file in candidates
            }

            # Rank candidates
            rankings = {
                "Static-only": sorted(
                    candidates,
                    key=lambda f: (
                        static_scores[f],
                        f
                    ),
                    reverse=True
                ),

                "Co-change-only": sorted(
                    candidates,
                    key=lambda f: (
                        cochange_scores[f],
                        f
                    ),
                    reverse=True
                ),

                "Hybrid": sorted(
                    candidates,
                    key=lambda f: (
                        hybrid_scores[f],
                        f
                    ),
                    reverse=True
                )
            }

            evaluated_queries += 1

            for model_name, ranking in rankings.items():

                for k in [5, 10]:

                    result = calculate_metrics(
                        ranking,
                        actual,
                        k
                    )

                    if result is not None:
                        metric_values[model_name][k].append(
                            result
                        )

    # Aggregate metrics
    summary = []

    for model_name, k_values in metric_values.items():

        for k, values in k_values.items():

            if not values:
                continue

            averages = np.mean(values, axis=0)

            summary.append({
                "model": model_name,
                "k": k,
                "precision": averages[0],
                "recall": averages[1],
                "f1_score": averages[2]
            })

    results = pd.DataFrame(summary)

    results.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n========== EVALUATION RESULTS ==========")

    print("Total prediction queries:", total_queries)
    print("Evaluated queries:", evaluated_queries)
    print("Skipped queries:", skipped_queries)

    print("\nCandidate files:", len(training_files))

    print("\nModel comparison:")

    print(
        results.to_string(index=False)
    )

    print(
        "\nResults saved to:",
        OUTPUT_DIR / "model_comparison.csv"
    )


if __name__ == "__main__":
    evaluate()