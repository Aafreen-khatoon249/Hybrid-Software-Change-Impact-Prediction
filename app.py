import streamlit as st
import pandas as pd
import networkx as nx
from pathlib import Path
import subprocess

st.set_page_config(page_title="Change Impact Intelligence", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

BASE = Path(__file__).parent
TRAIN_FILES = BASE / "dataset/temporal/train_commit_files.csv"
COCHANGE_FILE = BASE / "results/training_cochange_pairs.csv"
STATIC_FILE = BASE / "results/historical_static_dependencies_2019.csv"
RESULT_FILE = BASE / "results/evaluation/final_test_results.csv"

st.markdown("""
<style>
.block-container{padding-top:1.5rem;padding-bottom:2rem}
[data-testid="stSidebar"]{border-right:1px solid rgba(128,128,128,.2)}
.hero{padding:1.2rem 1.4rem;border:1px solid rgba(128,128,128,.25);
border-radius:14px;background:linear-gradient(110deg,rgba(70,110,180,.12),
rgba(40,160,140,.07));margin-bottom:1rem}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.22);
padding:12px 14px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return (pd.read_csv(TRAIN_FILES), pd.read_csv(COCHANGE_FILE),
            pd.read_csv(STATIC_FILE), pd.read_csv(RESULT_FILE))

# Git repository integration
REPO_PATH = BASE / "repositories" / "flask"


def run_git_command(command):
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


@st.cache_data(ttl=60)
def get_git_commits():
    command = [
        "git",
        "log",
        "--no-merges",
        "-20",
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


def get_commit_changed_files(commit_hash):
    command = [
        "git",
        "diff-tree",
        "--root",
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
            status = parts[0]

            if status.startswith("R") and len(parts) >= 3:
                file_name = parts[-1]
            else:
                file_name = parts[-1]

            if file_name.lower().endswith(".py"):
                changed_files.append(file_name)

    return sorted(set(changed_files))

def normalize(path):
    return str(path).replace("\\", "/").strip()

def find_column(df, names):
    return next((name for name in names if name in df.columns), None)

try:
    train, cochange, static, results = load_data()
except Exception as exc:
    st.error(f"Project data could not be loaded: {exc}")
    st.info("Keep app.py in the main project folder with dataset/ and results/ folders.")
    st.stop()

train_col = find_column(train, ["file_name", "file", "filename"])
c1 = find_column(cochange, ["file_1", "file1", "source_file"])
c2 = find_column(cochange, ["file_2", "file2", "target_file"])
cf = find_column(cochange, ["frequency", "cochange_count", "count", "weight"])
s1 = find_column(static, ["source_file", "file_1", "file1"])
s2 = find_column(static, ["target_file", "file_2", "file2"])

if not all([train_col, c1, c2, cf, s1, s2]):
    st.error("Required columns are missing from one of the project CSV files.")
    st.write("Training columns:", list(train.columns))
    st.write("Co-change columns:", list(cochange.columns))
    st.write("Static columns:", list(static.columns))
    st.stop()

candidates = sorted({normalize(x) for x in train[train_col].dropna()
                     if normalize(x).lower().endswith(".py")})

co_graph, static_graph = nx.Graph(), nx.Graph()
for _, row in cochange.iterrows():
    a, b = normalize(row[c1]), normalize(row[c2])
    try:
        weight = float(row[cf])
    except (TypeError, ValueError):
        continue
    if a and b and a != b:
        co_graph.add_edge(a, b, weight=weight)

for _, row in static.iterrows():
    a, b = normalize(row[s1]), normalize(row[s2])
    if a and b and a != b:
        static_graph.add_edge(a, b, weight=1.0)

def predict(changed_files, model):
    changed_files = [normalize(x) for x in changed_files]
    combined = {}
    for changed in changed_files:
        co_scores, static_scores = {}, {}
        if co_graph.has_node(changed):
            co_scores = {n: float(co_graph[changed][n].get("weight", 0))
                         for n in co_graph.neighbors(changed)}
        if static_graph.has_node(changed):
            static_scores = {n: 1.0 for n in static_graph.neighbors(changed)}
        max_co = max(co_scores.values()) if co_scores else 1.0
        for candidate in set(co_scores) | set(static_scores):
            if candidate in changed_files or candidate not in candidates:
                continue
            co_score = co_scores.get(candidate, 0.0) / max_co
            static_score = static_scores.get(candidate, 0.0)
            if model == "Static-only":
                score = static_score
            elif model == "Co-change-only":
                score = co_score
            else:
                score = 0.25 * static_score + 0.75 * co_score
            if score <= 0:
                continue
            item = combined.setdefault(candidate, {
                "Potentially Impacted File": candidate, "Prediction Score": 0.0,
                "Static Link": False, "Co-change Frequency": 0.0,
                "Evidence": set(), "Related to": set()
            })
            item["Prediction Score"] = max(item["Prediction Score"], score)
            item["Static Link"] = item["Static Link"] or static_score > 0
            item["Co-change Frequency"] = max(item["Co-change Frequency"],
                                                co_scores.get(candidate, 0.0))
            if static_score:
                item["Evidence"].add("Static dependency")
            if co_scores.get(candidate, 0) > 0:
                item["Evidence"].add("Historical co-change")
            item["Related to"].add(changed)
    rows = [{
        "Potentially Impacted File": x["Potentially Impacted File"],
        "Prediction Score": round(x["Prediction Score"], 4),
        "Static Link": "Yes" if x["Static Link"] else "No",
        "Co-change Frequency": int(x["Co-change Frequency"]),
        "Evidence": " + ".join(sorted(x["Evidence"])),
        "Related to": ", ".join(sorted(x["Related to"]))
    } for x in combined.values()]
    columns = ["Potentially Impacted File", "Prediction Score", "Static Link",
               "Co-change Frequency", "Evidence", "Related to"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["Prediction Score", "Co-change Frequency"], ascending=[False, False]
    ).reset_index(drop=True)

with st.sidebar:
    st.title("🔍 Change Impact")
    st.caption("Research prototype · Flask repository")
    st.divider()
    page = st.radio("NAVIGATION", [
    "Overview",
    "Impact Prediction",
    "Git Commit Prediction",
    "Repository Insights",
    "Model Evaluation",
    "Method & Limitations"
])
    st.divider()
    st.caption("**Research question**")
    st.caption("Does combining static dependencies and historical co-change "
               "information improve change-impact prediction?")
    st.caption("B.Tech CSE · Research Project")

if page == "Overview":
    st.markdown("""<div class="hero"><h2 style="margin:0">
    Software Change Impact Prediction</h2><p style="margin:.45rem 0 0">
    Explore relationships between changed files and rank other files that
    may need inspection or testing.</p></div>""", unsafe_allow_html=True)
    st.subheader("Repository at a glance")
    a, b, c, d = st.columns(4)
    a.metric("Training-known Python files", f"{len(candidates):,}")
    b.metric("Learned co-change pairs", f"{len(cochange):,}")
    c.metric("Historical static links", f"{len(static):,}")
    d.metric("Test commits", "767")
    st.subheader("Workspace")
    left, right = st.columns(2)
    with left:
        st.markdown("**Impact Prediction**")
        st.write("Select one or more changed files, choose a model and inspect "
                 "ranked recommendations with relationship evidence.")
        st.markdown("**Repository Insights**")
        st.write("Search known Python files and inspect learned co-change "
                 "pairs and static dependency links.")
    with right:
        st.markdown("**Model Evaluation**")
        st.write("Compare Static-only, Co-change-only and Hybrid on the "
                 "held-out chronological test set.")
        st.markdown("**Method & Limitations**")
        st.write("Review the experiment setup, prediction coverage and "
                 "known limitations.")
    st.info("Choose a workspace from the sidebar to get started.")

elif page == "Impact Prediction":
    st.title("Impact Prediction")
    st.caption("Select files changed in the current task. The system ranks "
               "other training-known Python files using learned relationships.")
    left, right = st.columns([2, 1])
    with left:
        selected_files = st.multiselect("Changed Python files", candidates,
            default=[], max_selections=10,
            placeholder="Search and select one or more files...")
    with right:
        model = st.selectbox("Prediction approach",
            ["Hybrid", "Co-change-only", "Static-only"])
        top_k = st.slider("Number of results to show", 5, 30, 10, 5)
    if model == "Hybrid":
        st.caption("Hybrid scoring: 25% static dependency + 75% co-change")
    if st.button("🔍 Predict potentially impacted files", type="primary",
                 use_container_width=True):
        if not selected_files:
            st.warning("Select at least one changed Python file first.")
        else:
            prediction = predict(selected_files, model)
            if prediction.empty:
                st.warning("No connected training-known candidate files were "
                           "found for this selection and model.")
            else:
                st.success(f"{len(prediction)} potentially impacted files found.")
                st.subheader(f"Top {min(top_k, len(prediction))} recommendations")
                st.dataframe(prediction.head(top_k), use_container_width=True,
                             hide_index=True)
                st.download_button("⬇️ Download complete prediction list",
                    prediction.to_csv(index=False).encode("utf-8"),
                    file_name="predicted_impacted_files.csv",
                    mime="text/csv")
                with st.expander("How to read these recommendations"):
                    st.write("**Prediction Score:** ranking score, not a probability.")
                    st.write("**Static Link:** import-based dependency link in "
                             "the historical snapshot.")
                    st.write("**Co-change Frequency:** training-history count "
                             "for the file pair.")
                    st.write("**Evidence:** relationship sources supporting "
                             "the recommendation.")
    st.divider()
    st.caption("Decision-support only: recommendations help identify files "
               "for code inspection or testing; they do not guarantee impact.")

elif page == "Git Commit Prediction":

    st.title("Git Commit-based Impact Prediction")
    st.caption(
        "Select a recent Git commit to automatically detect changed "
        "Python files and predict potentially impacted files."
    )

    try:
        commits = get_git_commits()

        if not commits:
            st.warning("No non-merge commits found in the repository.")

        else:
            selected_commit = st.selectbox(
                "Select a Git commit",
                commits,
                format_func=lambda c: (
                    f"{c['date']} | {c['message']} | {c['hash'][:10]}"
                )
            )

            st.markdown("### Selected Commit")

            a, b = st.columns(2)
            a.metric("Commit Date", selected_commit["date"])
            b.metric("Commit Hash", selected_commit["hash"][:12])

            st.write("**Commit message:**", selected_commit["message"])

            changed_files = get_commit_changed_files(
                selected_commit["hash"]
            )

            st.subheader("Automatically Detected Changed Files")

            if not changed_files:
                st.warning(
                    "No changed Python files were detected in this commit."
                )

            else:
                st.success(
                    f"{len(changed_files)} changed Python file(s) detected."
                )

                st.dataframe(
                    pd.DataFrame({
                        "Changed Python Files": changed_files
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                st.divider()
                st.subheader("Impact Prediction Settings")

                model = st.selectbox(
                    "Select prediction model",
                    ["Hybrid", "Co-change-only", "Static-only"],
                    key="git_prediction_model"
                )

                top_k = st.slider(
                    "Number of recommendations",
                    5, 30, 10, 5,
                    key="git_prediction_topk"
                )

                if model == "Hybrid":
                    st.caption(
                        "Hybrid scoring: 25% static dependency + "
                        "75% historical co-change."
                    )

                if st.button(
                    "Predict Impacted Files from Git Commit",
                    type="primary"
                ):

                    prediction = predict(changed_files, model)

                    if prediction.empty:
                        st.warning(
                            "No connected training-known candidate files "
                            "were found for this commit."
                        )
                    else:
                        st.success(
                            f"{len(prediction)} potentially impacted "
                            "files found."
                        )

                        st.subheader("Recommended Files")

                        st.dataframe(
                            prediction.head(top_k),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download Git-based Predictions",
                            prediction.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="git_commit_impact_predictions.csv",
                            mime="text/csv"
                        )

    except Exception as exc:
        st.error(f"Git analysis failed: {exc}")
elif page == "Repository Insights":
    st.title("Repository Insights")
    st.caption("Explore relationships learned from the training period.")
    a, b, c = st.columns(3)
    a.metric("Python candidate files", f"{len(candidates):,}")
    b.metric("Co-change relationships", f"{len(cochange):,}")
    c.metric("Static dependency relationships", f"{len(static):,}")
    query = st.text_input("Search training-known Python files",
                          placeholder="For example: flask/app.py")
    if query:
        matches = [f for f in candidates if query.lower() in f.lower()]
        st.write(f"{len(matches)} matching file(s)")
        st.dataframe(pd.DataFrame({"Python file": matches}),
                     use_container_width=True, hide_index=True)
    st.subheader("Most frequent co-change pairs")
    pairs = cochange.copy()
    pairs["File A"] = pairs[c1].map(normalize)
    pairs["File B"] = pairs[c2].map(normalize)
    pairs["Frequency"] = pd.to_numeric(pairs[cf], errors="coerce")
    pairs = pairs[["File A", "File B", "Frequency"]].dropna()
    st.dataframe(pairs.sort_values("Frequency", ascending=False).head(15),
                 use_container_width=True, hide_index=True)
    st.subheader("Historical static dependency links")
    links = static.copy()
    links["Source file"] = links[s1].map(normalize)
    links["Target file"] = links[s2].map(normalize)
    st.dataframe(links[["Source file", "Target file"]].drop_duplicates().head(25),
                 use_container_width=True, hide_index=True)

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    st.caption("Held-out chronological test-set results.")
    required = {"model", "k", "precision", "recall", "f1_score"}
    if not required.issubset(results.columns):
        st.error(f"Expected result columns: {sorted(required)}")
    else:
        display = results.copy()
        for metric in ["precision", "recall", "f1_score"]:
            display[metric] = pd.to_numeric(display[metric], errors="coerce")
        metric = st.selectbox("Compare metric",
            ["f1_score", "precision", "recall"],
            format_func=lambda x: {"f1_score": "F1-score",
                "precision": "Precision", "recall": "Recall"}[x])
        chart = display.pivot(index="model", columns="k", values=metric)
        st.bar_chart(chart)
        st.dataframe(display.sort_values(["k", "f1_score"],
            ascending=[True, False]), use_container_width=True, hide_index=True)
        st.subheader("Test-set coverage")
        x, y, z = st.columns(3)
        x.metric("Test queries", "636")
        y.metric("Evaluated queries", "393")
        z.metric("Skipped queries", "243")
        st.progress(393 / 636)
        st.caption("Query coverage: 61.8%")
        st.info("On this Flask dataset and experimental configuration, "
                "Co-change-only achieved the highest F1-score at K=5 and K=10. "
                "Hybrid was close but did not outperform it.")

elif page == "Method & Limitations":
    st.title("Method & Limitations")
    st.subheader("Experiment configuration")
    config = pd.DataFrame([
        ["Repository", "Pallets Flask"],
        ["Git history", "3,832 non-merge commits"],
        ["Chronological split", "70% training · 10% validation · 20% testing"],
        ["Static dependency snapshot", "6 May 2019"],
        ["Co-change patterns", "Training period only"],
        ["Hybrid weights", "25% static · 75% co-change"],
        ["Evaluation", "Precision@K · Recall@K · F1@K"]
    ], columns=["Setting", "Configuration"])
    st.dataframe(config, use_container_width=True, hide_index=True)
    st.subheader("Prediction approaches")
    st.markdown("- **Static-only:** import-based dependency relationship.\n"
                "- **Co-change-only:** normalized historical co-change frequency.\n"
                "- **Hybrid:** combines both using validation-selected weights "
                "(25% static and 75% co-change).")
    st.subheader("Coverage and limitations")
    st.markdown("- Candidate universe: 221 Python files known during training.\n"
        "- 393 of 636 test queries were evaluated (61.8% coverage).\n"
        "- 243 queries were skipped because no other changed file in the "
        "commit matched the training-known candidate universe.\n"
        "- Unseen files cannot be recommended by this candidate-based prototype.\n"
        "- Import-based static analysis misses some runtime, dynamic or indirect "
        "dependencies.\n"
        "- File moves and historical path changes can affect relationship matching.\n"
        "- Recommendations are decision-support hints, not proof of impact.")
    st.warning("The experiment did not show that Hybrid beats Co-change-only. "
               "This negative result is reported as observed.")

st.divider()
st.caption("Software Change Impact Prediction · Research prototype · "
           "Recommendations support inspection and testing.")
