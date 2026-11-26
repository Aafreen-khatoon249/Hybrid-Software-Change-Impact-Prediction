import pandas as pd
from itertools import combinations
from collections import Counter

# Load extracted Git file history
data = pd.read_csv("dataset/commit_files.csv")

# Keep Python files only
data = data[
    data["file_name"].str.endswith(".py", na=False)
]

# Store file pairs and their co-change frequency
pair_counter = Counter()

# Group files by commit
for commit_hash, group in data.groupby("commit_hash"):

    # Get unique Python files changed in this commit
    files = sorted(set(group["file_name"]))

    # Generate all possible file pairs
    for pair in combinations(files, 2):
        pair_counter[pair] += 1

# Convert results into a DataFrame
results = pd.DataFrame(
    [
        {
            "file_1": pair[0],
            "file_2": pair[1],
            "cochange_count": count
        }
        for pair, count in pair_counter.items()
    ]
)

# Sort by highest co-change frequency
results = results.sort_values(
    by="cochange_count",
    ascending=False
)

# Save results
results.to_csv(
    "results/cochange_pairs.csv",
    index=False
)

# Display analysis
print("\n========== CO-CHANGE ANALYSIS ==========")

print("Total Python file records:", len(data))

print(
    "Unique Python files:",
    data["file_name"].nunique()
)

print(
    "Commits containing Python files:",
    data["commit_hash"].nunique()
)

print(
    "Total co-change pairs:",
    len(results)
)

print("\nTop 20 frequently co-changed file pairs:")

print(results.head(20).to_string(index=False))

print("\nResults saved to: results/cochange_pairs.csv")