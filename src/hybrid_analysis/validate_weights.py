import pandas as pd
import numpy as np

from collections import defaultdict
from pathlib import Path


# Input datasets
TRAIN_FILE = "dataset/temporal/train_commit_files.csv"
VALIDATION_FILE = "dataset/temporal/validation_commit_files.csv"

COCHANGE_FILE = "results/training_cochange_pairs.csv"

# Static graph from 2019 training cutoff
STATIC_FILE = "results/historical_static_dependencies_2019.csv"

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

    hits = len(recommended & actual)

    precision = hits / k
    recall = hits / len(actual)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate():

    print("Starting validation-based weight selection...")

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)

    # Keep Python files only
    train = train[
        train["file_name"].str.endswith(".py", na=False)
    ].copy()

    validation = validation[
        validation["file_name"].str.endswith(".py", na=False)
    ].copy()

    train["file_name"] = normalize_paths(train["file_name"])
    validation["file_name"] = normalize_paths(
        validation["file_name"]
    )

    # Candidate files known during training
    training_files = set(train["file_name"].unique())

    cochange = load_cochange_patterns()
    static_graph = load_static_graph()

    # Weight combinations to compare
    weight_combinations = [
        (0.25, 0.75),
        (0.50, 0.50),
        (0.75, 0.25)
    ]

    metric_values = defaultdict(list)

    evaluated_queries = 0
    skipped_queries = 0
    total_queries = 0

    for commit_hash, group in validation.groupby("commit_hash"):

        changed_files = sorted(set(group["file_name"]))

        if len(changed_files) < 2:
            continue

        for source_file in changed_files:

            total_queries += 1

            actual_all = set(changed_files) - {source_file}

            # Only evaluate targets known during training
            actual = actual_all & training_files

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
                    (source_file, candidate),
                    0
                )

            # Normalize co-change scores
            max_score = max(cochange_scores.values(), default=0)

            if max_score > 0:
                normalized_cochange = {
                    file: score / max_score
                    for file, score in cochange_scores.items()
                }
            else:
                normalized_cochange = {
                    file: 0 for file in candidates
                }

            # Static-only and co-change-only rankings
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
                )
            }

            # Test different hybrid weights
            for static_weight, cochange_weight in weight_combinations:

                hybrid_scores = {
                    file: (
                        static_weight * static_scores[file]
                        + cochange_weight * normalized_cochange[file]
                    )
                    for file in candidates
                }

                hybrid_ranking = sorted(
                    candidates,
                    key=lambda f: (hybrid_scores[f], f),
                    reverse=True
                )

                model_name = (
                    f"Hybrid ({static_weight:.2f}:{cochange_weight:.2f})"
                )

                rankings[model_name] = hybrid_ranking

            evaluated_queries += 1

            for model_name, ranking in rankings.items():

                for k in [5, 10]:

                    result = calculate_metrics(
                        ranking,
                        actual,
                        k
                    )

                    metric_values[(model_name, k)].append(result)

    # Aggregate results
    summary = []

    for (model_name, k), values in metric_values.items():

        averages = np.mean(values, axis=0)

        summary.append({
            "model": model_name,
            "k": k,
            "precision": averages[0],
            "recall": averages[1],
            "f1_score": averages[2]
        })

    results = pd.DataFrame(summary)

    output_file = OUTPUT_DIR / "validation_weight_selection.csv"

    results.to_csv(output_file, index=False)

    print("\n========== VALIDATION RESULTS ==========")

    print("Total prediction queries:", total_queries)
    print("Evaluated queries:", evaluated_queries)
    print("Skipped queries:", skipped_queries)

    print("Candidate files:", len(training_files))

    print("\nModel comparison:")

    print(results.to_string(index=False))

    print("\nSaved to:", output_file)


if __name__ == "__main__":
    evaluate()