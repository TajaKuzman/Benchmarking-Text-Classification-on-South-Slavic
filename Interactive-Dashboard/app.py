
# app.py
import io
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import json

st.set_page_config(page_title="Model Leaderboard Explorer", layout="wide")

st.title("📊 Model Leaderboard Explorer")

# --------------------------
# Sidebar: data upload
# --------------------------

records = json.load(open("/home/tajak/Benchmarking-Text-Classification-on-South-Slavic/Genre-Automatic-Identification-Benchmark/results/results.json", "r"))


# Build long/tidy rows from nested structure
rows = []
for r in records:
    model = r.get("Model")
    test_ds = r.get("Test Dataset")

    # Optional overall scores (no specific language)
    for metric_key in ("Macro F1", "Micro F1"):
        if metric_key in r and r[metric_key] is not None:
            rows.append(
                {
                    "Model": model,
                    "Test Dataset": test_ds,
                    "Language": "(overall)",
                    "Metric": metric_key,
                    "Score": float(r[metric_key]),
                }
            )

    # Language-specific block
    lang_block = r.get("Language-Specific Scores") or {}
    if isinstance(lang_block, dict):
        for lang, scores in lang_block.items():
            if not isinstance(scores, dict):
                continue
            for metric_key in ("Macro F1", "Micro F1"):
                if metric_key in scores and scores[metric_key] is not None:
                    rows.append(
                        {
                            "Model": model,
                            "Test Dataset": test_ds,
                            "Language": lang,
                            "Metric": metric_key,
                            "Score": float(scores[metric_key]),
                        }
                    )

long_df = pd.DataFrame(rows)

# Clean types
long_df["Score"] = pd.to_numeric(long_df["Score"], errors="coerce")

# --------------------------
# Filters
# --------------------------
languages = sorted(long_df["Language"].dropna().unique().tolist())
metrics = sorted(long_df["Metric"].dropna().unique().tolist())
datasets = sorted(long_df["Test Dataset"].dropna().unique().tolist())

with st.sidebar:
    sel_langs = st.multiselect("Languages", options=languages, default=[l for l in languages if l != "(overall)"] or languages)
    sel_metrics = st.multiselect("Metrics", options=metrics, default=metrics)
    sel_datasets = st.multiselect("Test Datasets", options=datasets, default=datasets)
    model_search = st.text_input("Filter models (substring match)", value="")
    top_k = st.slider("Top K (after sorting)", min_value=3, max_value=max(3, len(long_df["Model"].unique())), value=min(10, len(long_df["Model"].unique())))

# Apply filters
filtered = long_df[
    long_df["Language"].isin(sel_langs) &
    long_df["Metric"].isin(sel_metrics) &
    long_df["Test Dataset"].isin(sel_datasets)
]
if model_search.strip():
    filtered = filtered[filtered["Model"].str.contains(model_search.strip(), case=False, na=False)]

if filtered.empty:
    st.warning("No rows match your current filters.")
    st.stop()

# Sort control: choose one (Language, Metric, Dataset) to define model ranking
combos_df = (
    filtered[["Language", "Metric", "Test Dataset"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["Language", "Metric", "Test Dataset"])
)
combos = list(combos_df.itertuples(index=False, name=None))  # -> list of (Language, Metric, Test Dataset)
combo_labels = [f"{L} · {M} · {D}" for (L, M, D) in combos]
sort_choice = st.selectbox("Sort models by:", options=combo_labels, index=0)
sort_lang, sort_metric, sort_ds = combos[combo_labels.index(sort_choice)]

sort_lang, sort_metric, sort_ds = combos[combo_labels.index(sort_choice)]
sort_slice = (
    filtered[
        (filtered["Language"] == sort_lang) &
        (filtered["Metric"] == sort_metric) &
        (filtered["Test Dataset"] == sort_ds)
    ].sort_values("Score", ascending=False)
)

if sort_slice.empty:
    st.warning("No rows match the selected sort combination. Try different filters.")
    st.stop()

# Keep only the top_k models by the chosen sort combo
top_models = sort_slice["Model"].head(top_k).tolist()
filtered_top = filtered[filtered["Model"].isin(top_models)].copy()
filtered_top["Lang/Metric"] = filtered_top["Language"] + " · " + filtered_top["Metric"]
filtered_top["Model (meta)"] = filtered_top.apply(
    lambda r: f"{r['Model']}", axis=1
)

# Consistent ordering
model_cat = pd.CategoricalDtype(categories=top_models, ordered=True)
filtered_top["Model"] = filtered_top["Model"].astype(model_cat)

# --------------------------
# Main chart
# --------------------------
st.subheader("Interactive comparison")
title = f"Models ranked by {sort_lang} · {sort_metric} · {sort_ds}"
fig = px.bar(
    filtered_top,
    x="Score",
    y="Model",
    color="Lang/Metric",
    barmode="group" if len(sel_langs) * len(sel_metrics) <= 2 else "relative",
    hover_data=["Test Dataset", "Language", "Metric"],
    title=title,
    height=560,
)
fig.update_layout(xaxis_title="Score", yaxis_title="", legend_title="Language · Metric")
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Breakdown tabs
# --------------------------
st.markdown("#### Breakdown by dataset and language/metric")
group_keys = (
    filtered_top[["Test Dataset", "Lang/Metric"]]
    .drop_duplicates()
    .itertuples(index=False, name=None)  # yields tuples (dataset, lang_metric)
)
group_keys = list(group_keys)

labels = [f"{d} · {lm}" for (d, lm) in group_keys]
if labels:
    tabs = st.tabs(labels)
    for tab, (d, lm) in zip(tabs, group_keys):
        with tab:
            sub = (
                filtered_top[(filtered_top["Test Dataset"] == d) & (filtered_top["Lang/Metric"] == lm)]
                .sort_values("Score", ascending=False)
            )
            fig2 = px.bar(sub, x="Score", y="Model", title=f"{d} · {lm}", height=420)
            fig2.update_layout(xaxis_title="Score", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(
                sub[["Model", "Test Dataset", "Language", "Metric", "Score"]],
                use_container_width=True, hide_index=True
            )
else:
    st.info("No breakdown tabs for the current view.")
