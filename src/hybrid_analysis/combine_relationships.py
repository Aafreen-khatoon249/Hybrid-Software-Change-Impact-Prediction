import pandas as pd
from pathlib import Path

# Input files
COCHANGE_FILE = "results/cochange_pairs.csv"
DEPENDENCY_FILE = "results/static_dependencies.csv"

# Output file
OUTPUT_FILE = Path("results/hybrid_relationships.csv")


def combine_relationships():

    print("Starting Hybrid Relationship Analysis...")

    # Load both datasets
    cochange = pd.read_csv(COCHANGE_FILE)
    dependencies = pd.read_csv(DEPENDENCY_FILE)

    # Normalize file paths
    for column in ["file_1", "file_2"]:
        cochange[column] = (
            cochange[column]
            .astype(str)
            .str.replace("\\", "/", regex=False)
        )

    for column in ["source_file", "target_file"]:
        dependencies[column] = (
            dependencies[column]
            .astype(str)
            .str.replace("\\", "/", regex=False)
        )

    # Create unordered file pairs for static dependencies
    dependencies["file_1"] = dependencies[
        ["source_file", "target_file"]
    ].min(axis=1)

    dependencies["file_2"] = dependencies[
        ["source_file", "target_file"]
    ].max(axis=1)

    # Remove duplicate pairs
    static_pairs = dependencies[
        ["file_1", "file_2"]
    ].drop_duplicates()

    static_pairs["static_dependency"] = 1

    # Combine co-change and static dependency information
    hybrid = pd.merge(
        cochange,
        static_pairs,
        on=["file_1", "file_2"],
        how="outer"
    )

    # Fill missing values
    hybrid["cochange_count"] = (
        hybrid["cochange_count"].fillna(0).astype(int)
    )

    hybrid["static_dependency"] = (
        hybrid["static_dependency"].fillna(0).astype(int)
    )

    # Sort by co-change frequency
    hybrid = hybrid.sort_values(
        by=["static_dependency", "cochange_count"],
        ascending=False
    )

    # Save combined dataset
    hybrid.to_csv(OUTPUT_FILE, index=False)

    # Display summary
    print("\n========== HYBRID ANALYSIS RESULTS ==========")

    print("Total combined file pairs:", len(hybrid))

    print(
        "Pairs with static dependency:",
        hybrid["static_dependency"].sum()
    )

    print(
        "Pairs with co-change history:",
        (hybrid["cochange_count"] > 0).sum()
    )

    print(
        "Pairs having both relationships:",
        (
            (hybrid["static_dependency"] == 1)
            & (hybrid["cochange_count"] > 0)
        ).sum()
    )

    print("\nTop 20 combined relationships:")

    print(hybrid.head(20).to_string(index=False))

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    combine_relationships()