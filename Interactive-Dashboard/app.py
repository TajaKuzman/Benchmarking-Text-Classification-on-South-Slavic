import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


directory = os.getcwd()
print(directory)

plt.rcParams["figure.dpi"] = 500


# ---------------------
# Data loading
# ---------------------
@st.cache_data
def load_data(path: str = "Interactive-Dashboard/results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
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
    if not isinstance(metric, str):
        return str(metric)
    metric = metric.replace("-", "_")
    parts = metric.split("_")
    return " ".join(p.capitalize() if p.lower() != "f1" else "F1" for p in parts)


def language_task_metric_summary(df: pd.DataFrame) -> list[str]:
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


def compute_model_rankings(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    if df.empty:
        return (
            pd.DataFrame(
                columns=[
                    "Model",
                    "Average ranking",
                    "Average ranking English",
                    "Average ranking South Slavic",
                ]
            ),
            {},
            [],
        )

    df_all = df.copy()
    tasks_all = sorted(df_all["task"].unique())
    n_tasks = len(tasks_all)

    tasks_per_model = df_all.groupby("model")["task"].nunique()
    full_models = tasks_per_model[tasks_per_model == n_tasks].index.tolist()

    if not full_models:
        return (
            pd.DataFrame(
                columns=[
                    "Model",
                    "Average ranking",
                    "Average ranking English",
                    "Average ranking South Slavic",
                ]
            ),
            {},
            [],
        )

    df_rank = df_all[df_all["model"].isin(full_models)].copy()

    agg = (
        df_rank.groupby(["task", "language", "model"])["value"]
        .mean()
        .reset_index(name="score")
    )

    if agg.empty:
        return (
            pd.DataFrame(
                columns=[
                    "Model",
                    "Average ranking",
                    "Average ranking English",
                    "Average ranking South Slavic",
                ]
            ),
            {},
            full_models,
        )

    agg["rank"] = agg.groupby(["task", "language"])["score"].rank(
        ascending=False, method="average"
    )

    overall_rank = agg.groupby("model")["rank"].mean().rename("Average ranking")

    per_lang_rank = (
        agg.groupby(["model", "language"])["rank"]
        .mean()
        .reset_index()
    )

    languages_all = sorted(per_lang_rank["language"].unique())
    english_langs = [
        l for l in languages_all
        if l.strip().lower() in {"english", "en"}
    ]

    if english_langs:
        english_rank = (
            per_lang_rank[per_lang_rank["language"].isin(english_langs)]
            .groupby("model")["rank"]
            .mean()
            .rename("Average ranking English")
        )
    else:
        english_rank = pd.Series(
            index=overall_rank.index,
            data=float("nan"),
            name="Average ranking English",
        )

    if english_langs:
        south_slavic_rank = (
            per_lang_rank[~per_lang_rank["language"].isin(english_langs)]
            .groupby("model")["rank"]
            .mean()
            .rename("Average ranking South Slavic")
        )
    else:
        south_slavic_rank = overall_rank.rename("Average ranking South Slavic")

    lang_pivot = (
        per_lang_rank.pivot(index="model", columns="language", values="rank")
        if not per_lang_rank.empty
        else pd.DataFrame()
    )

    rank_df = overall_rank.to_frame().join(english_rank, how="left").join(
        south_slavic_rank, how="left"
    )

    if not lang_pivot.empty:
        rank_df = rank_df.join(lang_pivot, how="left")

    rank_df = rank_df.reset_index().rename(columns={"model": "Model"})

    base_cols = ["Model", "Average ranking", "Average ranking English", "Average ranking South Slavic"]
    lang_cols = [c for c in rank_df.columns if c not in base_cols]
    rank_df = rank_df[base_cols + lang_cols]

    rank_df = rank_df.sort_values("Average ranking", ascending=True).reset_index(drop=True)

    tasks_per_lang_series = (
        agg.groupby("language")["task"]
        .apply(lambda s: sorted(s.unique()))
    )
    tasks_per_lang = tasks_per_lang_series.to_dict()

    return rank_df, tasks_per_lang, full_models


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


def plot_task_heatmap(df_task: pd.DataFrame, metric_label: str):
    if df_task.empty:
        return

    matrix = df_task.pivot_table(
        index="model",
        columns="language",
        values="value",
        aggfunc="mean",
    )

    if matrix.empty:
        return

    matrix["Average"] = matrix.mean(axis=1)
    data = matrix.sort_values(by="Average", ascending=False)

    # Smaller figure
    plt.figure(figsize=(8, 6), dpi=500)

    ax = sns.heatmap(
        data,
        annot=False,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": metric_label},
    )

    n_rows, n_cols = data.shape
    for x in range(n_cols):
        col = data.iloc[:, x]
        max_val = col.max()
        for y in range(n_rows):
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
                    fontsize=7,
                )
            else:
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    st.pyplot(plt.gcf())
    plt.close()


def plot_task_bar_chart(df_task: pd.DataFrame, metric_label: str):
    if df_task.empty:
        return

    matrix = df_task.pivot_table(
        index="language",
        columns="model",
        values="value",
        aggfunc="mean",
    )

    if matrix.empty:
        return

    # Smallest score in the table
    data_min = float(np.nanmin(matrix.values))
    ymin = round(data_min - 0.01, 2)

    plt.figure(figsize=(4, 3))  # smaller figure

    ax = matrix.plot(kind="bar", width=0.9, colormap="tab20")

    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        title="Model",
    )

    ax.set_ylim(ymin, None)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel(metric_label)

    ymin, ymax = ax.get_ylim()
    ymin = max(0.0, round(ymin, 2))
    ymax = round(ymax + 0.05, 2)
    if ymax > ymin:
        ticks = np.arange(ymin, ymax + 1e-9, 0.05)
        ax.set_yticks(ticks)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="gray", alpha=0.5)
    ax.set_axisbelow(True)

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
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            color: black !important;
        }

        p, span, div {
            font-size: 18px !important;
        }

        .stSidebar, .stSidebar p, .stSidebar span, .stSidebar label {
            font-size: 18px !important;
            font-family: 'Inter', sans-serif !important;
            color: black !important;
        }

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

        .stDataFrame div, .stDataFrame table, .stDataFrame th, .stDataFrame td {
            font-size: 16px !important;
            font-family: 'Inter', sans-serif !important;
            color: black !important;
        }

        [data-baseweb="tag"] {
            font-size: 16px !important;
            font-family: 'Inter', sans-serif !important;
        }

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

    st.markdown(
        """
Use the sections below to:
- See a global comparison of models based on average ranking across all tasks and languages.
- Explore detailed per-task, per-language, per-model results with interactive filters.
        """
    )

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

    st.markdown("---")

    # ---------------------
    # Global summary
    # ---------------------
    st.subheader("Global summary")

    st.markdown(
        """
Models are ranked separately for each task–language pair based on their average score
(across all available metrics). Ranks (1 = best) are then averaged across all tasks and
languages. Only models evaluated on all tasks are included.
        """
    )

    ranking_df, tasks_per_lang, full_models = compute_model_rankings(df)

    if ranking_df.empty:
        st.info("No global ranking can be computed: no model was evaluated on all tasks.")
    else:
        chart_rank = (
            alt.Chart(ranking_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Model:N",
                    sort=alt.EncodingSortField(
                        field="Average ranking", order="ascending"
                    ),
                    title="Model",
                ),
                y=alt.Y(
                    "Average ranking:Q",
                    title="Average ranking (lower is better)",
                ),
                tooltip=list(ranking_df.columns),
            )
        )
        st.altair_chart(chart_rank, use_container_width=True)

        base_cols = ["Model", "Average ranking", "Average ranking English", "Average ranking South Slavic"]
        lang_cols = [c for c in ranking_df.columns if c not in base_cols]

        if lang_cols:
            selected_rank_langs = st.multiselect(
                "Languages to display in ranking table",
                options=lang_cols,
                default=lang_cols,
            )
        else:
            selected_rank_langs = []

        display_cols = base_cols + selected_rank_langs

        st.dataframe(
            ranking_df[display_cols].style.format(
                {col: "{:.3f}" for col in ranking_df.columns if col != "Model"}
            )
        )

        if full_models:
            st.caption(
                "Models included in the global ranking (tested on all tasks): "
                + ", ".join(full_models)
            )

        if tasks_per_lang:
            st.markdown("Tasks included in the ranking per language:")
            for lang in sorted(tasks_per_lang.keys()):
                tasks_str = ", ".join(tasks_per_lang[lang])
                st.caption(f"{lang}: {tasks_str}")

    st.markdown("---")

    # ---------------------
    # Task-level view
    # ---------------------
    st.subheader("Task-level view")

    st.markdown(
        "Use the filters below to inspect detailed per-task, per-language, per-model results."
    )

    with st.expander(
        "Task coverage: for each task, which metrics are available",
        expanded=False,
    ):
        if not task_summary_lines:
            st.info("No data available to summarise task coverage.")
        else:
            for line in task_summary_lines:
                st.write(line)

    tasks = sorted(df["task"].unique())
    metrics = sorted(df["metric"].unique())
    languages_all = sorted(df["language"].unique())
    models_all = sorted(df["model"].unique())

    selected_tasks = st.multiselect(
        "Tasks",
        options=tasks,
        default=tasks,
    )

    selected_metric = st.selectbox(
        "Metric", options=metrics, index=0
    )

    selected_languages = st.multiselect(
        "Languages / Dialects",
        options=languages_all,
        default=languages_all,
    )

    selected_models = st.multiselect(
        "Models",
        options=models_all,
        default=models_all,
    )

    metric_label = format_metric_name(selected_metric)

    if not selected_tasks:
        effective_tasks = tasks
    else:
        effective_tasks = selected_tasks

    filtered = apply_filters(
        df,
        task=None,
        languages=selected_languages,
        models=selected_models,
        metric=selected_metric,
    )
    filtered = filtered[filtered["task"].isin(effective_tasks)]

    if filtered.empty:
        st.warning("No rows match your current filters.")
        return

    if len(effective_tasks) != 1:
        st.info(
            "Select exactly one task in the 'Tasks' selector above to see the detailed "
            "task-specific table and bar chart. Heatmaps and raw tables below "
            "reflect all selected tasks."
        )

    # Detailed task-specific table (matrix) when exactly one task is chosen
    if len(effective_tasks) == 1:
        task_name = effective_tasks[0]
        filtered_task = filtered[filtered["task"] == task_name]

        st.markdown(
            f"Results for task: {task_name} "
            f"({metric_label}, filtered by models/languages)."
        )

        matrix = (
            filtered_task.pivot_table(
                index="model",
                columns="language",
                values="value",
                aggfunc="mean",
            )
        )
        if not matrix.empty:
            matrix["Average"] = matrix.mean(axis=1)
            matrix = matrix.sort_values("Average", ascending=False)

            # Turn index into a column and name it "Model"
            matrix = matrix.reset_index().rename(columns={"model": "Model"})

            # Reorder columns: Model, languages, Average
            col_order = ["Model"] + [
                c for c in matrix.columns if c not in ["Model", "Average"]
            ] + ["Average"]
            matrix = matrix[col_order]

            st.markdown(f"Metric shown in the table: {metric_label}")
            st.dataframe(matrix.style.format("{:.3f}"))

    # Heatmaps, bar plots, and tables per task
    st.markdown("Heatmaps, bar plots, and tables for each task in the current selection:")

    tasks_in_filtered = sorted(filtered["task"].unique())
    for t in tasks_in_filtered:
        df_t = filtered[filtered["task"] == t]
        st.markdown(f"Task: {t}")

        # Plots side-by-side (each ~1/2 page width)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Heatmap (models × languages):")
            plot_task_heatmap(df_t, metric_label)
        with col2:
            st.markdown("Bar plot (languages on X axis, models in legend):")
            plot_task_bar_chart(df_t, metric_label)

        # Task-specific table directly below the plots
        st.markdown(f"Results table for task {t}")

        df_t_mm = df_t

        # Average over multiple runs per (model, language, metric)
        grouped = (
            df_t_mm.groupby(["model", "language", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Pivot: rows = model, columns = (language, metric)
        table = grouped.pivot_table(
            index="model",
            columns=["language", "metric"],
            values="value",
            aggfunc="mean",
        )

        # Normalize metric name for checking
        selected_metric_norm = selected_metric.lower().replace("-", "_")

        # Try to find columns matching selected metric
        metric_cols = [
            col for col in table.columns
            if isinstance(col, tuple)
            and selected_metric_norm in str(col[1]).lower().replace("-", "_")
        ]

        avg_col_name = "Average"

        # Compute Average across selected columns (if any)
        if metric_cols:
            table[avg_col_name] = table[metric_cols].mean(axis=1)
        else:
            # Fall back: mean over all numeric columns if nothing matched
            table[avg_col_name] = table.mean(axis=1, numeric_only=True)

        # Flatten column names, making sure Average does not become "Average ()"
        flat_cols = []
        for col in table.columns:
            # Special case: keep Average as a simple column name
            if (
                col == avg_col_name
                or (isinstance(col, tuple) and col[0] == avg_col_name)
            ):
                flat_cols.append(avg_col_name)
                continue

            if isinstance(col, tuple):
                lang, met = col
                flat_cols.append(f"{lang} ({format_metric_name(met)})")
            else:
                flat_cols.append(col)
        table.columns = flat_cols

        # Bring model index into a column, if present
        table = table.reset_index()

        # Make sure we actually have the model column, then rename it
        if "model" in table.columns:
            table = table.rename(columns={"model": "Model"})
        elif "index" in table.columns:
            # In case the index name was lost and reset_index created "index"
            table = table.rename(columns={"index": "Model"})

        # Reorder columns: Model | other | Average, but only keep existing ones
        desired_order = []

        if "Model" in table.columns:
            desired_order.append("Model")

        # All columns except Model and Average, in their current order
        other_cols = [
            c for c in table.columns
            if c not in ["Model", avg_col_name]
        ]
        desired_order.extend(other_cols)

        if avg_col_name in table.columns:
            desired_order.append(avg_col_name)

        # Keep only columns that actually exist, to avoid KeyError
        desired_order = [c for c in desired_order if c in table.columns]
        table = table[desired_order]

        # Sort by Average (descending) if it exists
        if avg_col_name in table.columns:
            table = table.sort_values(avg_col_name, ascending=False)

        # Format numeric columns
        num_cols = table.select_dtypes(include=["float", "int"]).columns
        styler = table.style.format("{:.6f}", subset=num_cols)

        st.dataframe(styler)


if __name__ == "__main__":
    main()
