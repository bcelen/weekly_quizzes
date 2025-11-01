import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weekly Quiz Adjustment", layout="wide")

st.title("📊 Weekly Quiz Adjustment Dashboard")

# --- User Input ---
week_labels = [f"Week {i} Quiz results" for i in range(1, 7)]
week_files = [f"week{i}.csv" for i in range(1, 7)]

week_selection = st.selectbox("Select a quiz week:", options=week_labels)
week_index = week_labels.index(week_selection)
filename = week_files[week_index]

# Slider for mean selection
target_mean = st.slider("🎯 Adjusted Average (Target)", min_value=7.4, max_value=7.6, value=7.5, step=0.1)

# --- Download Raw CSV from GitHub ---
github_url = f"https://raw.githubusercontent.com/bcelen/weekly_quizzes/main/{filename}"

try:
    df = pd.read_csv(github_url)
    raw_scores = df.iloc[:, 0].dropna()
    raw_scores = pd.to_numeric(raw_scores, errors='coerce')
    raw_scores = raw_scores.dropna()
    raw_scores = np.clip(raw_scores.values, 0, 10)

    st.success(f"✅ Loaded {filename} with {len(raw_scores)} valid scores.")

    # --- Compute Z Scores ---
    z_scores = (raw_scores - np.mean(raw_scores)) / np.std(raw_scores)

    # --- Find std dev to cap % > 8 at 30% ---
    z_threshold = norm.ppf(1 - 0.30)
    required_std = (8 - target_mean) / z_threshold

    adjusted_scores = np.clip(z_scores * required_std + target_mean, 0, 10)

    # --- Summary Statistics ---
    pct_above_8 = np.mean(adjusted_scores >= 8) * 100
    summary = {
        "📦 Students": len(raw_scores),
        "🎯 Target Mean": round(target_mean, 2),
        "📐 Adjusted Mean": round(np.mean(adjusted_scores), 2),
        "📊 Adjusted Std Dev": round(np.std(adjusted_scores), 2),
        "🔥 % ≥ 8.0": f"{pct_above_8:.1f}%"
    }

    st.subheader("Summary")
    st.dataframe(pd.DataFrame([summary]))

    # --- Histogram ---
    st.subheader("Distribution of Adjusted Scores")
    bins = [0, 5, 6, 7, 8, 9, 10]
    labels = ["F", "P", "H3", "H2", "H1", "H1+"]
    colors = ["#dddddd", "#bbbbee", "#88aaff", "#5588ff", "#0044cc", "#002288"]
    hist, edges = np.histogram(adjusted_scores, bins=np.linspace(0, 10, 21))

    fig, ax = plt.subplots()
    for i in range(len(hist)):
        left = edges[i]
        right = edges[i+1]
        color = "#0044cc" if right >= 8 else "#bbbbbb"
        ax.bar(left, hist[i], width=right-left, color=color, align="edge", edgecolor="black")

    ax.set_title("Adjusted Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # --- Table Output ---
    st.subheader("Raw vs Adjusted Scores")
    table_df = pd.DataFrame({
        "Original Score": np.round(raw_scores, 2),
        "Adjusted Score": np.round(adjusted_scores, 2)
    })
    st.dataframe(table_df, use_container_width=True)

    # --- Download Option ---
    csv = table_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Adjusted Scores CSV", data=csv, file_name=f"{filename.replace('.csv', '')}_adjusted.csv")

except Exception:
    st.warning("⚠️ The quiz marks are not available yet. Please check back later.")
