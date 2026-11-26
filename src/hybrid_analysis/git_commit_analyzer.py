import subprocess
from pathlib import Path


# Project folder automatically detect karo
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Flask Git repository ka path
REPO_PATH = PROJECT_ROOT / "repositories" / "flask"


def run_git_command(command):
    """Git command execute karke output return karta hai."""

    result = subprocess.run(
        command,
        cwd=REPO_PATH,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


def get_recent_commits(limit=10):
    """Repository ke recent commits fetch karta hai."""

    command = [
    "git",
    "log",
    "--no-merges",
    f"-{limit}",
    "--format=%H%x1f%ad%x1f%s",
    "--date=short"
    ]
    output = run_git_command(command)

    commits = []

    for line in output.splitlines():

        parts = line.split("\x1f", 2)

        if len(parts) == 3:
            commits.append({
                "hash": parts[0],
                "date": parts[1],
                "message": parts[2]
            })

    return commits


def get_changed_files(commit_hash):
    """Ek commit mein changed files detect karta hai."""

    command = [
        "git",
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        commit_hash
    ]

    output = run_git_command(command)

    changed_files = []

    for line in output.splitlines():

        parts = line.split("\t")

        if len(parts) >= 2:
            changed_files.append({
                "change_type": parts[0],
                "file_name": parts[-1]
            })

    return changed_files


def main():

    print("\n====================================")
    print(" GIT COMMIT CHANGE ANALYZER")
    print("====================================")

    commits = get_recent_commits(10)

    print(f"\nTotal commits displayed: {len(commits)}")

    for index, commit in enumerate(commits, start=1):

        print("\n------------------------------------")
        print(f"Commit {index}")
        print(f"Hash: {commit['hash'][:12]}")
        print(f"Date: {commit['date']}")
        print(f"Message: {commit['message']}")

        files = get_changed_files(commit["hash"])

        print(f"\nChanged files: {len(files)}")

        for file in files:
            print(
                f"  [{file['change_type']}] "
                f"{file['file_name']}"
            )

    print("\n====================================")
    print(" ANALYSIS COMPLETED")
    print("====================================")


if __name__ == "__main__":
    main()