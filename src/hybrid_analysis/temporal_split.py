import pandas as pd
from pathlib import Path

# Input files
COMMITS_FILE = "dataset/commits.csv"
FILES_FILE = "dataset/commit_files.csv"

# Output directory
OUTPUT_DIR = Path("dataset/temporal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_temporal_split():

    print("Creating chronological 70/10/20 split...")

    # Load datasets
    commits = pd.read_csv(COMMITS_FILE)
    files = pd.read_csv(FILES_FILE)

    # Parse commit dates
    commits["commit_date"] = pd.to_datetime(
        commits["commit_date"],
        utc=True
    )

    # Sort commits from oldest to newest
    commits = commits.sort_values(
        "commit_date"
    ).reset_index(drop=True)

    # Calculate chronological split indexes
    total_commits = len(commits)

    train_end = int(total_commits * 0.70)
    validation_end = int(total_commits * 0.80)

    # Divide commits into three periods
    train_commits = commits.iloc[:train_end].copy()

    validation_commits = commits.iloc[
        train_end:validation_end
    ].copy()

    test_commits = commits.iloc[
        validation_end:
    ].copy()

    # Separate file changes using commit hashes
    train_files = files[
        files["commit_hash"].isin(
            train_commits["commit_hash"]
        )
    ].copy()

    validation_files = files[
        files["commit_hash"].isin(
            validation_commits["commit_hash"]
        )
    ].copy()

    test_files = files[
        files["commit_hash"].isin(
            test_commits["commit_hash"]
        )
    ].copy()

    # Save datasets
    datasets = {
        "train_commits.csv": train_commits,
        "validation_commits.csv": validation_commits,
        "test_commits.csv": test_commits,
        "train_commit_files.csv": train_files,
        "validation_commit_files.csv": validation_files,
        "test_commit_files.csv": test_files
    }

    for filename, dataframe in datasets.items():
        dataframe.to_csv(
            OUTPUT_DIR / filename,
            index=False
        )

    # Display results
    print("\n========== TEMPORAL SPLIT RESULTS ==========")

    print("Total commits:", total_commits)

    print("Training commits:", len(train_commits))
    print("Validation commits:", len(validation_commits))
    print("Testing commits:", len(test_commits))

    print(
        "\nTraining period:",
        train_commits["commit_date"].min(),
        "to",
        train_commits["commit_date"].max()
    )

    print(
        "\nValidation period:",
        validation_commits["commit_date"].min(),
        "to",
        validation_commits["commit_date"].max()
    )

    print(
        "\nTesting period:",
        test_commits["commit_date"].min(),
        "to",
        test_commits["commit_date"].max()
    )

    print("\nSaved datasets in:", OUTPUT_DIR)


if __name__ == "__main__":
    create_temporal_split()