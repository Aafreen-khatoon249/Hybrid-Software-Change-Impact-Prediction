# Hybrid Static-Dependency and Co-Change Learning for Software Change Impact Prediction

## Project Overview

This project explores software change impact prediction using a combination of static dependency analysis and historical co-change relationships.

The system analyzes source-code dependencies and Git commit history to rank files that may be impacted when a software file is changed.

## Research Objective

To investigate whether combining static dependencies and historical co-change learning improves software change impact prediction compared with using either technique independently.

## Methodology

The project uses three approaches:

1. **Static Dependency Analysis:** Identifies relationships between Python files using import-based dependencies.
2. **Co-Change Learning:** Learns relationships between files that were modified together in historical Git commits.
3. **Hybrid Approach:** Combines static dependency and co-change scores using a weighted ranking method.

A chronological train-validation-test split is used to reduce future-data leakage.

## Dataset

The experiments use the Git history of the [Pallets Flask repository](https://github.com/pallets/flask).

The extracted history contains:
- 3,832 non-merge commits
- 9,322 file-change records
- 643 unique file paths

## Experimental Results

The final test evaluation includes 393 evaluated queries and 636 prediction queries, with 61.8% coverage.

| Model | Precision@10 | Recall@10 | F1@10 |
|---|---:|---:|---:|
| Static-only | 0.0601 | 0.1560 | 0.0763 |
| Co-change-only | 0.0855 | 0.2560 | 0.1129 |
| Hybrid (25:75) | 0.0827 | 0.2489 | 0.1094 |

### Key Finding

The co-change-only approach achieved the strongest overall test performance. The hybrid approach performed closely, while the static-only approach showed weaker results.

This indicates that historical co-change information was particularly useful for this dataset. Combining both techniques did not outperform co-change learning alone.

## Technologies Used

- Python
- Git
- PyDriller
- Pandas
- Streamlit
- Matplotlib
- Static dependency analysis

## Project Structure

```text
Software-Change-Impact-Prediction/
├── dataset/
├── models/
├── results/
│   ├── evaluation/
│   └── visualizations/
├── src/
│   ├── data_extraction/
│   ├── dependency_analysis/
│   └── hybrid_analysis/
├── app.py
├── requirements.txt
└── README.md
```

## Run the Application

1. Clone this repository.
2. Open the project folder in VS Code.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit dashboard:

```bash
streamlit run app.py
```

## Dashboard Features

- Project overview
- Single-file impact prediction
- Multiple-file impact prediction
- Git commit prediction
- Repository insights
- Model evaluation
- Methodology and limitations

## Limitations

- The experiments use a single software repository.
- Static dependency analysis is based on Python import relationships.
- Co-change relationships indicate historical association, not necessarily causation.
- Prediction coverage is limited by the training candidate set.
- The system provides decision support, not guaranteed impact predictions.

## Conclusion

This project presents an experimental comparison of static dependency analysis, historical co-change learning, and a hybrid approach for software change impact prediction. The results highlight the usefulness of historical co-change patterns and identify opportunities for improving hybrid prediction methods.