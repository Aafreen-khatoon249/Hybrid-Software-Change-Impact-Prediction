import pandas as pd
import numpy as np

from collections import defaultdict
from pathlib import Path


TRAIN_FILE = "dataset/temporal/train_commit_files.csv"
TEST_FILE = "dataset/temporal/test_commit_files.csv"

COCHANGE_FILE = "results/training_cochange_pairs.csv"

STATIC_FILE = "results/historical_static_dependencies_2019.csv"

OUTPUT_DIR = Path("results/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_paths(series):
    return (
        series.astype(str)
        .str.replace("\\", "/", regex=False)
    )


def load_static_graph():

    data = pd.read_csv(STATIC_FILE)

    data["source_file"] = normalize_paths(data["source_file"])
    data["target_file"] = normalize_paths(data["target_file"])

    graph = defaultdict(set)

    for _, row in data.iterrows():

        source = row["source_file"]
        target = row["target_file"]

        graph[source].add(target)
        graph[target].add(source)

    return graph


def load_cochange():

    data = pd.read_csv(COCHANGE_FILE)

    data["file_1"] = normalize_paths(data["file_1"])
    data["file_2"] = normalize_paths(data["file_2"])

    patterns = defaultdict(int)

    for _, row in data.iterrows():

        patterns[(row["file_1"], row["file_2"])] = row["cochange_count"]
        patterns[(row["file_2"], row["file_1"])] = row["cochange_count"]

    return patterns


def calculate_metrics(ranking, actual, k):

    recommended = set(ranking[:k])

    hits = len(recommended & actual)

    precision = hits / k
    recall = hits / len(actual)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate():

    print("Starting Final Test Evaluation...")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    train = train[
        train["file_name"].str.endswith(".py", na=False)
    ].copy()

    test = test[
        test["file_name"].str.endswith(".py", na=False)
    ].copy()

    train["file_name"] = normalize_paths(train["file_name"])
    test["file_name"] = normalize_paths(test["file_name"])

    training_files = set(train["file_name"].unique())

    static_graph = load_static_graph()
    cochange = load_cochange()

    models = [
        "Static-only",
        "Co-change-only",
        "Hybrid (25:75)"
    ]

    metrics = {
        model: {5: [], 10: []}
        for model in models
    }

    total_queries = 0
    evaluated_queries = 0
    skipped_queries = 0

    for commit_hash, group in test.groupby("commit_hash"):

        changed_files = sorted(set(group["file_name"]))

        if len(changed_files) < 2:
            continue

        for source_file in changed_files:

            total_queries += 1

            actual = (
                set(changed_files) - {source_file}
            ) & training_files

            if not actual:
                skipped_queries += 1
                continue

            candidates = training_files - {source_file}

            static_scores = {}
            cochange_scores = {}

            for candidate in candidates:

                static_scores[candidate] = (
                    1 if candidate in static_graph[source_file]
                    else 0
                )

                cochange_scores[candidate] = cochange.get(
                    (source_file, candidate), 0
                )

            max_score = max(cochange_scores.values(), default=0)

            if max_score > 0:

                normalized = {
                    f: score / max_score
                    for f, score in cochange_scores.items()
                }

            else:

                normalized = {
                    f: 0 for f in candidates
                }

            hybrid_scores = {
                f: (
                    0.25 * static_scores[f]
                    + 0.75 * normalized[f]
                )
                for f in candidates
            }

            rankings = {

                "Static-only": sorted(
                    candidates,
                    key=lambda f: (static_scores[f], f),
                    reverse=True
                ),

                "Co-change-only": sorted(
                    candidates,
                    key=lambda f: (cochange_scores[f], f),
                    reverse=True
                ),

                "Hybrid (25:75)": sorted(
                    candidates,
                    key=lambda f: (hybrid_scores[f], f),
                    reverse=True
                )
            }

            evaluated_queries += 1

            for model, ranking in rankings.items():

                for k in [5, 10]:

                    result = calculate_metrics(
                        ranking,
                        actual,
                        k
                    )

                    metrics[model][k].append(result)

    summary = []

    for model, k_values in metrics.items():

        for k, values in k_values.items():

            averages = np.mean(values, axis=0)

            summary.append({

                "model": model,
                "k": k,
                "precision": averages[0],
                "recall": averages[1],
                "f1_score": averages[2]

            })

    results = pd.DataFrame(summary)

    output_file = OUTPUT_DIR / "final_test_results.csv"

    results.to_csv(output_file, index=False)

    print("\n========== FINAL TEST RESULTS ==========")

    print("Total queries:", total_queries)
    print("Evaluated queries:", evaluated_queries)
    print("Skipped queries:", skipped_queries)

    print("\nModel comparison:")

    print(results.to_string(index=False))

    print("\nSaved to:", output_file)


if __name__ == "__main__":
    evaluate()