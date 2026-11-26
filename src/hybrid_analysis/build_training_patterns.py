import pandas as pd
from itertools import combinations
from collections import Counter
from pathlib import Path

# Training dataset
TRAIN_FILE = "dataset/temporal/train_commit_files.csv"

# Output
OUTPUT_FILE = Path("results/training_cochange_pairs.csv")


def build_training_patterns():

    print("Learning co-change patterns from training data...")

    data = pd.read_csv(TRAIN_FILE)

    # Keep Python files only
    data = data[
        data["file_name"].str.endswith(".py", na=False)
    ].copy()

    # Normalize file paths
    data["file_name"] = (
        data["file_name"]
        .str.replace("\\", "/", regex=False)
    )

    pair_counter = Counter()

    # Group files by commit
    for commit_hash, group in data.groupby("commit_hash"):

        files = sorted(set(group["file_name"]))

        # Generate unique file pairs
        for pair in combinations(files, 2):
            pair_counter[pair] += 1

    # Convert patterns into DataFrame
    results = pd.DataFrame(
        [
            {
                "file_1": pair[0],
                "file_2": pair[1],
                "cochange_count": count
            }
            for pair, count in pair_counter.items()
        ],
        columns=["file_1", "file_2", "cochange_count"]
    )

    if not results.empty:
        results = results.sort_values(
            "cochange_count",
            ascending=False
        )

    # Save training patterns
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    results.to_csv(OUTPUT_FILE, index=False)

    print("\n========== TRAINING PATTERNS ==========")

    print("Python file-change records:", len(data))

    print(
        "Unique Python files:",
        data["file_name"].nunique()
    )

    print(
        "Training commits containing Python files:",
        data["commit_hash"].nunique()
    )

    print("Learned co-change pairs:", len(results))

    print("\nTop 10 learned relationships:")

    if not results.empty:
        print(results.head(10).to_string(index=False))

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    build_training_patterns()