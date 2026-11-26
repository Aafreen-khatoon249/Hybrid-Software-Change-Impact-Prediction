import csv
from pathlib import Path
from pydriller import Repository

# Project repository location
REPO_PATH = "repositories/flask"

# Output folder
OUTPUT_DIR = Path("dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# CSV file paths
COMMITS_FILE = OUTPUT_DIR / "commits.csv"
FILES_FILE = OUTPUT_DIR / "commit_files.csv"


def extract_git_history():

    commits_data = []
    files_data = []

    print("Starting Git history extraction...")

    # Read all commits from Flask repository
    for commit in Repository(REPO_PATH).traverse_commits():

        # Skip merge commits for consistent file-change records
        if commit.merge:
            continue

        commits_data.append({
            "commit_hash": commit.hash,
            "commit_date": commit.author_date.isoformat(),
            "author": commit.author.name,
            "message": commit.msg
        })

        # Extract files modified in this commit
        for modified_file in commit.modified_files:

            files_data.append({
                "commit_hash": commit.hash,
                "file_name": modified_file.new_path
                    or modified_file.old_path,
                "change_type": modified_file.change_type.name
            })

    # Save commit information
    with open(COMMITS_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "commit_hash",
                "commit_date",
                "author",
                "message"
            ]
        )

        writer.writeheader()
        writer.writerows(commits_data)

    # Save modified file information
    with open(FILES_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "commit_hash",
                "file_name",
                "change_type"
            ]
        )

        writer.writeheader()
        writer.writerows(files_data)

    print("\nGit history extraction completed!")
    print("Total commits extracted:", len(commits_data))
    print("Total file-change records:", len(files_data))
    print("Saved files:")
    print(COMMITS_FILE)
    print(FILES_FILE)


if __name__ == "__main__":
    extract_git_history()