import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load final evaluation results
RESULT_FILE = "results/evaluation/final_test_results.csv"

OUTPUT_DIR = Path("results/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(RESULT_FILE)

# Select results for K = 10
data = data[data["k"] == 10].copy()

metrics = ["precision", "recall", "f1_score"]

# Create separate chart for each metric
for metric in metrics:

    plt.figure(figsize=(9, 5))

    plt.bar(
        data["model"],
        data[metric]
    )

    plt.title(
        f"Model Comparison - {metric.upper()}@10"
    )

    plt.xlabel("Prediction Model")
    plt.ylabel(metric.upper())

    plt.xticks(rotation=15)

    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{metric}_comparison.png"

    plt.savefig(output_file, dpi=300)

    plt.close()

    print("Saved:", output_file)


print("\nAll visualization charts generated successfully!")