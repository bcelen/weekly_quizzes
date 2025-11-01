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

with st.container():
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        week_selection = st.selectbox("Select a quiz week:", options=week_labels)
    with col2:
        target_mean = st.slider("🎯 Adjusted Average (Target)", min_value=7.4, max_value=7.6, value=7.5, step=0.01)
    with col3:
        target_pct_above_8 = st.slider("🔥 Max % of Adjusted Marks ≥ 8.0", min_value=0.20, max_value=0.30, value=0.30, step=0.01)

week_index = week_labels.index(week_selection)
filename = week_files[week_index]

# --- Download Raw CSV from GitHub ---
github_url = f"https://raw.githubusercontent.com/bcelen/weekly_quizzes/main/{filename}"

try:
    df = pd.read_csv(github_url)
    raw_marks = df.iloc[:, 0].dropna()
    raw_marks = pd.to_numeric(raw_marks, errors='coerce')
    raw_marks = raw_marks.dropna()
    raw_marks = np.clip(raw_marks.values, 0, 10)

    st.success(f"✅ Loaded {filename} with {len(raw_marks)} valid marks.")

    # --- Compute Z Scores ---
    z_scores = (raw_marks - np.mean(raw_marks)) / np.std(raw_marks)

    # --- Find std dev to cap % > 8 at user-specified level ---
    z_threshold = norm.ppf(1 - target_pct_above_8)
    required_std = (8 - target_mean) / z_threshold

    adjusted_marks = np.clip(z_scores * required_std + target_mean, 0, 10)

    # --- Sort by original marks ---
    sorted_indices = np.argsort(raw_marks)
    raw_sorted = raw_marks[sorted_indices]
    adjusted_sorted = adjusted_marks[sorted_indices]

    # --- Summary Statistics ---
    actual_summary = {
        "📦 Students": len(raw_marks),
        "🎯 Mean": round(np.mean(raw_marks), 2),
        "📐 Std Dev": round(np.std(raw_marks), 2),
        "🔥 % ≥ 8.0": f"{np.mean(raw_marks >= 8) * 100:.1f}%"
    }

    adjusted_summary = {
        "📦 Students": len(adjusted_marks),
        "🎯 Mean": round(np.mean(adjusted_marks), 2),
        "📐 Std Dev": round(np.std(adjusted_marks), 2),
        "🔥 % ≥ 8.0": f"{np.mean(adjusted_marks >= 8) * 100:.1f}%"
    }

    summary_df = pd.DataFrame([actual_summary, adjusted_summary], index=["Actual Marks", "Adjusted Marks"])

    st.subheader("Summary")
    st.dataframe(summary_df)

    # --- Personal Mark Lookup ---
    st.subheader("🔍 Find Your Adjusted Mark and Rank")
    with st.form("lookup_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_mark = st.number_input("Enter your actual quiz mark (0-10):", min_value=0.0, max_value=10.0, step=0.1)
        submitted = st.form_submit_button("Find My Adjusted Mark")

    if submitted:
        user_z = (user_mark - np.mean(raw_marks)) / np.std(raw_marks)
        user_adjusted = round(np.clip(user_z * required_std + target_mean, 0, 10), 2)
        rank = int(np.sum(adjusted_marks > user_adjusted)) + 1
        total = len(adjusted_marks)

        with col2:
            st.markdown(f"""
                **Your adjusted mark is:** `{user_adjusted}`  
                **Your rank is:** `{rank}` out of `{total}` students.
            """)

        # --- Add marker to plot ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(raw_sorted)), raw_sorted, marker='o', linestyle='-', label='Original', color='gray')
        ax.plot(range(len(adjusted_sorted)), adjusted_sorted, marker='o', linestyle='-', label='Adjusted', color='blue')
        ax.axhline(user_adjusted, color='red', linestyle='--', linewidth=1, label='Your Adjusted Mark')
        ax.axhline(user_mark, color='orange', linestyle='--', linewidth=1, label='Your Original Mark')
        ax.set_ylim(0, 10)
        ax.set_xlabel("Student Index (Sorted by Original Mark)")
        ax.set_ylabel("Mark")
        ax.set_title("Student Marks: Raw vs Adjusted with Your Mark Highlighted")
        ax.legend()
        st.pyplot(fig)

    else:
        # --- Line Graph ---
        st.subheader("📈 Student Marks (Raw vs Adjusted, Sorted by Raw Marks)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(raw_sorted)), raw_sorted, marker='o', linestyle='-', label='Original', color='gray')
        ax.plot(range(len(adjusted_sorted)), adjusted_sorted, marker='o', linestyle='-', label='Adjusted', color='blue')
        ax.set_ylim(0, 10)
        ax.set_xlabel("Student Index (Sorted by Original Mark)")
        ax.set_ylabel("Mark")
        ax.set_title("Student Marks: Raw vs Adjusted")
        ax.legend()
        st.pyplot(fig)

except Exception:
    st.warning("⚠️ The quiz marks are not available yet. Please check back later.")
