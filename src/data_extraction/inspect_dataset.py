import pandas as pd

# Load both CSV files
commits = pd.read_csv("dataset/commits.csv")
commit_files = pd.read_csv("dataset/commit_files.csv")


print("\n========== COMMITS DATASET ==========")

print("\nTotal rows and columns:")
print(commits.shape)

print("\nColumn names:")
print(commits.columns.tolist())

print("\nFirst 5 commits:")
print(commits.head())

print("\nMissing values:")
print(commits.isnull().sum())

print("\nCommit date range:")
print("Oldest:", commits["commit_date"].min())
print("Newest:", commits["commit_date"].max())


print("\n========== COMMIT FILES DATASET ==========")

print("\nTotal rows and columns:")
print(commit_files.shape)

print("\nColumn names:")
print(commit_files.columns.tolist())

print("\nFirst 10 file changes:")
print(commit_files.head(10))

print("\nMissing values:")
print(commit_files.isnull().sum())

print("\nUnique commits in file dataset:")
print(commit_files["commit_hash"].nunique())

print("\nUnique files:")
print(commit_files["file_name"].nunique())

print("\nChange types:")
print(commit_files["change_type"].value_counts())


print("\n========== ANALYSIS COMPLETED ==========")