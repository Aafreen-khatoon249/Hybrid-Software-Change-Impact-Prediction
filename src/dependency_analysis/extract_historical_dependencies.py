import ast
import csv
from pathlib import Path
from collections import Counter

# Flask repository location
REPO_PATH = Path("repositories/flask_2019")

# Output file
OUTPUT_FILE = Path("results/historical_static_dependencies_2019.csv")
OUTPUT_FILE.parent.mkdir(exist_ok=True)


def get_module_name(file_path):
    """
    Convert Python file paths into module names.
    Example: src/flask/helpers.py -> src.flask.helpers
    """

    relative_path = file_path.relative_to(REPO_PATH)

    parts = list(relative_path.with_suffix("").parts)

    if parts[-1] == "__init__":
        parts.pop()

    return ".".join(parts)


def find_internal_module(module_name, module_map):
    """
    Find an existing internal Python module.
    Also handles imports of parent packages.
    """

    while module_name:

        if module_name in module_map:
            return module_map[module_name]

        if "." not in module_name:
            break

        module_name = module_name.rsplit(".", 1)[0]

    return None


def extract_dependencies():

    print("Starting Static Dependency Analysis...")

    # Find all Python files
    python_files = list(REPO_PATH.rglob("*.py"))

    # Create module-name to file-path mapping
    module_map = {}

    for file_path in python_files:

        module_name = get_module_name(file_path)
        module_map[module_name] = file_path

    dependencies = []
    syntax_errors = 0

    # Analyse each Python file
    for file_path in python_files:

        source_module = get_module_name(file_path)

        try:
            source_code = file_path.read_text(
                encoding="utf-8"
            )

            tree = ast.parse(source_code)

        except (SyntaxError, UnicodeDecodeError, OSError):
            syntax_errors += 1
            continue

        # Determine the current package for relative imports
        current_package = source_module

        if file_path.name != "__init__.py":
            current_package = source_module.rpartition(".")[0]

        # Find import statements
        for node in ast.walk(tree):

            imported_modules = []

            # Example: import flask.helpers
            if isinstance(node, ast.Import):

                for alias in node.names:
                    imported_modules.append(alias.name)

            # Example: from flask import helpers
            elif isinstance(node, ast.ImportFrom):

                if node.level == 0:
                    base_module = node.module or ""

                else:
                    package_parts = current_package.split(".")

                    remove_count = node.level - 1

                    if remove_count >= len(package_parts):
                        continue

                    if remove_count:
                        package_parts = package_parts[:-remove_count]

                    base_module = ".".join(package_parts)

                    if node.module:
                        base_module = (
                            f"{base_module}.{node.module}"
                            if base_module
                            else node.module
                        )

                if base_module:
                    imported_modules.append(base_module)

                # Check whether imported names are submodules
                for alias in node.names:

                    if alias.name == "*":
                        continue

                    if base_module:
                        imported_modules.append(
                            f"{base_module}.{alias.name}"
                        )

            # Resolve imports to actual internal files
            for imported_module in imported_modules:

                target_path = find_internal_module(
                    imported_module,
                    module_map
                )

                if target_path is None:
                    continue

                target_module = get_module_name(target_path)

                if source_module == target_module:
                    continue

                dependencies.append({
                    "source_file": str(
                        file_path.relative_to(REPO_PATH)
                    ).replace("\\", "/"),

                    "target_file": str(
                        target_path.relative_to(REPO_PATH)
                    ).replace("\\", "/"),

                    "imported_module": imported_module
                })

    # Remove duplicate dependency records
    unique_dependencies = {
        (
            item["source_file"],
            item["target_file"],
            item["imported_module"]
        ): item
        for item in dependencies
    }

    dependencies = list(unique_dependencies.values())

    # Save dependency records
    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_file",
                "target_file",
                "imported_module"
            ]
        )

        writer.writeheader()
        writer.writerows(dependencies)

    # Display results
    source_counts = Counter(
        item["source_file"]
        for item in dependencies
    )

    print("\n========== STATIC DEPENDENCY RESULTS ==========")

    print("Python files discovered:", len(python_files))

    print("Files with parsing errors:", syntax_errors)

    print("Internal dependencies found:", len(dependencies))

    print(
        "Files having at least one dependency:",
        len(source_counts)
    )

    print("\nTop 10 files with most dependencies:")

    for file_name, count in source_counts.most_common(10):
        print(file_name, "->", count)

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    extract_dependencies()