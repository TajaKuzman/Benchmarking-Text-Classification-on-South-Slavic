import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt 
import os


directory = os.getcwd()
print(directory)


# ---------------------
# Data loading
# ---------------------
@st.cache_data
def load_data(path: str = "Interactive-Dashboard/results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"task", "model", "language", "metric", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"results.csv is missing required columns: {', '.join(sorted(missing))}"
        )
    return df


# ---------------------
# Helper functions
# ---------------------
def format_metric_name(metric: str) -> str:
    """
    Make metric names a bit nicer for display.
    Example: 'micro_f1' -> 'Micro F1'
    """
    if not isinstance(metric, str):
        return str(metric)
    metric = metric.replace("-", "_")
    parts = metric.split("_")
    return " ".join(p.capitalize() if p.lower() != "f1" else "F1" for p in parts)


def language_task_metric_summary(df: pd.DataFrame) -> list[str]:
    """
    Build text summary per language:
    Slovenian: PIQA (Accuracy); COPA (Accuracy); Sentiment (Micro F1, Macro F1)
    """
    if df.empty:
        return []

    grouped = (
        df.groupby(["language", "task"])["metric"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )

    lang_map: dict[str, dict[str, list[str]]] = {}
    for _, row in grouped.iterrows():
        lang = row["language"]
        task = row["task"]
        metrics = [format_metric_name(m) for m in row["metric"]]

        lang_map.setdefault(lang, {})[task] = metrics

    lines: list[str] = []
    for lang in sorted(lang_map.keys()):
        task_parts = []
        for task in sorted(lang_map[lang].keys()):
            metrics_str = ", ".join(lang_map[lang][task])
            task_parts.append(f"{task} ({metrics_str})")
        if task_parts:
            lines.append(f"{lang}: " + "; ".join(task_parts))
    return lines


def task_metric_summary(df: pd.DataFrame) -> list[str]:
    """
    Build text summary per task:
    PIQA: Accuracy
    Sentiment: Micro F1, Macro F1
    """
    if df.empty:
        return []

    grouped = (
        df.groupby("task")["metric"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )

    lines: list[str] = []
    for _, row in grouped.iterrows():
        task = row["task"]
        metrics = [format_metric_name(m) for m in row["metric"]]
        metrics_str = ", ".join(metrics)
        lines.append(f"{task}: {metrics_str}")
    return lines


def overall_model_averages(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Compute average score per model for the given metric, but
    only for models that were evaluated on all tasks that have this metric.

    Returns:
      model_avg_df, tasks_included, models_included, models_excluded
    """
    metric_df = df[df["metric"] == metric].copy()
    if metric_df.empty:
        return pd.DataFrame(columns=["model", "avg_score"]), [], [], []

    tasks_included = sorted(metric_df["task"].unique())
    n_tasks = len(tasks_included)

    # Count for each model how many distinct tasks it has for this metric
    counts = metric_df.groupby("model")["task"].nunique()
    full_models = counts[counts == n_tasks].index.tolist()
    excluded_models = counts[counts < n_tasks].index.tolist()

    if not full_models:
        return pd.DataFrame(columns=["model", "avg_score"]), tasks_included, [], excluded_models

    filtered = metric_df[metric_df["model"].isin(full_models)]

    model_avg = (
        filtered.groupby("model")["value"]
        .mean()
        .reset_index(name="avg_score")
        .sort_values("avg_score", ascending=False)
    )

    return model_avg, tasks_included, full_models, excluded_models


def apply_filters(
    df: pd.DataFrame,
    task: str | None,
    languages: list[str],
    models: list[str],
    metric: str,
) -> pd.DataFrame:
    data = df.copy()
    if task and task != "[All tasks]":
        data = data[data["task"] == task]

    if languages:
        data = data[data["language"].isin(languages)]

    if models:
        data = data[data["model"].isin(models)]

    if metric:
        data = data[data["metric"] == metric]

    return data


def sort_dataframe(df: pd.DataFrame, sort_choice: str) -> pd.DataFrame:
    if sort_choice.startswith("Score"):
        return df.sort_values("value", ascending=False)
    elif sort_choice.startswith("Model"):
        return df.sort_values(["model", "language"])
    elif sort_choice.startswith("Language"):
        return df.sort_values(["language", "model"])
    else:
        return df


def plot_task_heatmap(df_task: pd.DataFrame, metric_label: str):
    """
    For a single task subset (df_task), create a heatmap with:
    - Y axis: models
    - X axis: languages (+ an 'Average' column)
    - Values: metric scores
    Uses seaborn + matplotlib, with bold max per column.
    """
    if df_task.empty:
        return
    plt.rcParams['text.antialiased'] = True
    plt.rcParams['lines.antialiased'] = True

    # Pivot: rows=models, columns=languages, values=metric value
    matrix = df_task.pivot_table(
        index="model",
        columns="language",
        values="value",
        aggfunc="mean",
    )

    if matrix.empty:
        return

    # Add row-wise average column
    matrix["Average"] = matrix.mean(axis=1)

    # Sort models by average descending
    data = matrix.sort_values(by="Average", ascending=False)

    # Create figure
    plt.figure(figsize=(10, 7), dpi=500)

    # Heatmap
    ax = sns.heatmap(
        data,
        annot=False,      # custom annotations below
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5,         # adjust if needed for your metric range
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": metric_label},
    )

    # Add custom text annotations, bolding max per column
    n_rows, n_cols = data.shape
    for x in range(n_cols):  # columns
        col = data.iloc[:, x]
        max_val = col.max()
        for y in range(n_rows):  # rows
            val = col.iloc[y]
            if pd.isna(val):
                continue
            text = f"{val:.3f}"
            if val == max_val:
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=8,
                )
            else:
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    st.pyplot(plt.gcf())
    plt.close()


# ---------------------
# Streamlit layout
# ---------------------
def main():
    st.set_page_config(
        page_title="CLASSLA LLM Evaluation Dashboard for South Slavic Languages",
        layout="wide",
    )

    st.markdown(
    """
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

        <style>

        /* Apply your font to everything */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            color: black !important;
        }

        /* Default text size for paragraphs, st.write, st.markdown, etc */
        p, span, div {
            font-size: 18px !important;
        }

        /* Sidebar text */
        .stSidebar, .stSidebar p, .stSidebar span, .stSidebar label {
            font-size: 18px !important;
            font-family: 'Inter', sans-serif !important;
            color: black !important;
        }

        /* Headers */
        h1 {
            font-size: 38px !important;
            font-weight: 600 !important;
            color: black !important;
        }
        h2 {
            font-size: 30px !important;
            font-weight: 500 !important;
            color: black !important;
        }
        h3 {
            font-size: 24px !important;
            font-weight: 500 !important;
            color: black !important;
        }

        /* Tables (dataframes) */
        .stDataFrame div, .stDataFrame table, .stDataFrame th, .stDataFrame td {
            font-size: 16px !important;
            font-family: 'Inter', sans-serif !important;
            color: black !important;
        }

        /* Multiselect tags / selected items */
        [data-baseweb="tag"] {
            font-size: 16px !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Selectbox dropdown text */
        [data-baseweb="select"] * {
            font-family: 'Inter', sans-serif !important;
            font-size: 18px !important;
        }
        [data-baseweb="tag"] {
            background-color: #0080aa !important;
            color: white !important;
        }
        [data-baseweb="tag"]:hover {
            background-color: #006c8d !important;
            color: white !important;
        }
        [data-baseweb="select"] div[data-baseweb="option"]:hover {
            background-color: #0080aa20 !important;
        }
        html, body, [class*="css"] {
            color: black !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.image("Interactive-Dashboard/CLASSLA-k-centre-transparent-background.png", width='stretch')

    st.title("CLASSLA LLM Evaluation Dashboard for South Slavic Languages")

    try:
        df = load_data()
    except Exception as e:
        st.error(
            "Could not load results.csv. Make sure it exists next to app.py "
            "and has columns: task, model, language, metric, value."
        )
        st.exception(e)
        return

    # Short instructions for users
    st.markdown(
        """
Use the filters in the left sidebar to:
- Select a subset of tasks, metrics, languages, and models.
- Explore global averages across all tasks and languages.
- Inspect detailed per-task, per-language, per-model results.
        """
    )

    # Coverage summaries at the top of the page
    lang_summary_lines = language_task_metric_summary(df)
    task_summary_lines = task_metric_summary(df)

    with st.expander(
        "Language coverage: for each language, which tasks and metrics are available",
        expanded=False,
    ):
        if not lang_summary_lines:
            st.info("No data available to summarise language coverage.")
        else:
            for line in lang_summary_lines:
                st.write(line)

    with st.expander(
        "Task coverage: for each task, which metrics are available",
        expanded=False,
    ):
        if not task_summary_lines:
            st.info("No data available to summarise task coverage.")
        else:
            for line in task_summary_lines:
                st.write(line)

    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("Filters")

    tasks = sorted(df["task"].unique())
    metrics = sorted(df["metric"].unique())
    languages = sorted(df["language"].unique())
    models = sorted(df["model"].unique())

    selected_task = st.sidebar.selectbox(
        "Task", options=["[All tasks]"] + tasks, index=0
    )

    selected_metric = st.sidebar.selectbox(
        "Metric", options=metrics, index=0
    )

    selected_languages = st.sidebar.multiselect(
        "Languages / Dialects",
        options=languages,
        default=languages,
    )

    selected_models = st.sidebar.multiselect(
        "Models",
        options=models,
        default=models,
    )

    sort_choice = st.sidebar.selectbox(
        "Sort rows by",
        options=[
            "Score (descending)",
            "Model (A–Z)",
            "Language (A–Z)",
        ],
    )

    # ---------------------
    # Main dashboard: global summary
    # ---------------------
    st.subheader("Global summary")

    metric_label = format_metric_name(selected_metric)
    st.markdown(
        f"Average {metric_label} of each model across all tasks and languages."
    )

    model_avg, tasks_included, models_included, models_excluded = overall_model_averages(
        df, selected_metric
    )

    if model_avg.empty:
        st.info(
            "No model has been evaluated on all tasks for this metric, "
            "so no global model average can be shown."
        )
    else:
        chart = (
            alt.Chart(model_avg)
            .mark_bar()
            .encode(
                x=alt.X("model:N", sort="-y"),
                y=alt.Y("avg_score:Q"),
                tooltip=["model", "avg_score"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(model_avg.reset_index(drop=True))

        tasks_str = ", ".join(tasks_included) if tasks_included else "none"
        models_included_str = ", ".join(models_included) if models_included else "none"

        st.caption(
            f"Only models that were evaluated on all tasks for this metric are included "
            f"in the global average. Tasks: {tasks_str}. Models included: {models_included_str}."
        )

        if models_excluded:
            models_excluded_str = ", ".join(sorted(models_excluded))
            st.caption(f"Models excluded due to missing tasks for this metric: {models_excluded_str}")

    st.markdown("---")

    # ---------------------
    # Per-task dashboard
    # ---------------------
    st.subheader("Task-level view")

    if selected_task == "[All tasks]":
        st.info(
            "Choose a specific task in the sidebar to see more detailed "
            "per-model, per-language results for that task. "
            "Below you can see heatmaps for all tasks included by your current filters."
        )
        filtered = apply_filters(
            df,
            task=None,
            languages=selected_languages,
            models=selected_models,
            metric=selected_metric,
        )
    else:
        filtered = apply_filters(
            df,
            task=selected_task,
            languages=selected_languages,
            models=selected_models,
            metric=selected_metric,
        )

    if filtered.empty:
        st.warning("No rows match your current filters.")
        return

    # Sort table
    filtered = sort_dataframe(filtered, sort_choice)

    # Detailed matrix + bar chart only when a single task is chosen
    if selected_task != "[All tasks]":
        st.markdown(
            f"Results for task: {selected_task} "
            f"({metric_label}, filtered by models/languages)."
        )

        matrix = (
            filtered.pivot_table(
                index="model",
                columns="language",
                values="value",
                aggfunc="mean",
            )
            .sort_index()
        )
        st.dataframe(matrix.style.format("{:.3f}"))

        chart = (
            alt.Chart(filtered)
            .mark_bar()
            .encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("value:Q", title=metric_label),
                color=alt.Color("language:N", title="Language"),
                tooltip=["task", "model", "language", "metric", "value"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # Heatmaps for each task in the filtered data
    st.markdown("Heatmaps (models × languages) for each task in the current selection:")

    tasks_in_filtered = sorted(filtered["task"].unique())
    for t in tasks_in_filtered:
        df_t = filtered[filtered["task"] == t]
        st.markdown(f"Task: {t}")
        plot_task_heatmap(df_t, metric_label)

    # Raw table
    st.markdown("Raw rows (after filters and sorting):")
    st.dataframe(
        filtered[["task", "model", "language", "metric", "value"]]
        .reset_index(drop=True)
        .style.format({"value": "{:.3f}"})
    )


if __name__ == "__main__":
    main()
