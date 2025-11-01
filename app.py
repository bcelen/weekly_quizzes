import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Weekly Quiz Results Dashboard", layout="wide")

st.title("📊 Weekly Quiz Results Dashboard")

# --- User Input ---
week_labels = [f"Week {i} Quiz results" for i in range(1, 7)]
week_files = [f"week{i}.csv" for i in range(1, 7)]

with st.container():
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        week_selection = st.selectbox("📅 Select a quiz week:", options=week_labels)
    with col2:
        target_mean = st.slider(
            "🧮 Adjusted Average",
            min_value=7.4, max_value=7.6, value=7.5, step=0.01,
            help="This sets the average of the adjusted marks to meet MBS grading policy."
        )
    with col3:
        target_pct_above_8 = st.slider(
            "🎯 Maximum percentage of adjusted marks above 8",
            min_value=0.20, max_value=0.30, value=0.30, step=0.01,
            help="Adjusts standard deviation to ensure at most this percentage of adjusted marks are 8.0 or higher."
        )

st.markdown(
    """
    > MBS grade policy requires that the **mean of the final marks (out of 100)** be between **74 and 76**, and that the percentage of marks classified as **H1** does not exceed **30%**.  
    > This dashboard applies an adjustment to the quiz marks to reflect these requirements **while preserving the original z-scores**.  
    > Please note: although these distributions provide an indication of final outcomes, they may change by the end of the subject, particularly due to **zero marks that might be revised after the consensus date**.
    """
)

week_index = week_labels.index(week_selection)
filename = week_files[week_index]

# --- Download Raw CSV from GitHub ---
github_url = f"https://raw.githubusercontent.com/bcelen/weekly_quizzes/main/{filename}"

try:
    df = pd.read_csv(github_url)
    original_marks = df.iloc[:, 0].dropna()
    original_marks = pd.to_numeric(original_marks, errors='coerce')
    original_marks = original_marks.dropna()
    original_marks = np.clip(original_marks.values, 0, 10)

    # --- Compute Z Scores ---
    z_scores = (original_marks - np.mean(original_marks)) / np.std(original_marks)

    # --- Find std dev to cap % > 8 at user-specified level ---
    z_threshold = norm.ppf(1 - target_pct_above_8)
    required_std = (8 - target_mean) / z_threshold

    adjusted_marks = np.clip(z_scores * required_std + target_mean, 0, 10)

    # --- Sort by original marks ---
    sorted_indices = np.argsort(original_marks)
    original_sorted = original_marks[sorted_indices]
    adjusted_sorted = adjusted_marks[sorted_indices]

    # --- Summary Statistics ---
    original_summary = {
        "📊 Students": len(original_marks),
        "🎯 Mean": f"{np.mean(original_marks):.2f}",
        "📐 Std Dev": f"{np.std(original_marks):.2f}"
    }

    adjusted_summary = {
        "📊 Students": len(adjusted_marks),
        "🎯 Mean": f"{np.mean(adjusted_marks):.2f}",
        "📐 Std Dev": f"{np.std(adjusted_marks):.2f}"
    }

    summary_df = pd.DataFrame([original_summary, adjusted_summary], index=["Original Marks", "Adjusted Marks"])

    # --- Summary and Student Lookup ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Summary")
        st.dataframe(summary_df)

    with col2:
        st.subheader("🔍 Find Your Adjusted Mark and Rank")
        with st.form("lookup_form"):
            input_col, result_col = st.columns(2)
            with input_col:
                user_mark = st.number_input(
                    "Enter your original quiz mark (0-10):",
                    min_value=0.0, max_value=10.0, step=0.1,
                    help="Enter the mark you received to find your adjusted mark and ranking."
                )
            submitted = st.form_submit_button("Find My Adjusted Mark")

        if submitted:
            user_z = (user_mark - np.mean(original_marks)) / np.std(original_marks)
            user_adjusted = round(np.clip(user_z * required_std + target_mean, 0, 10), 2)
            rank = int(np.sum(adjusted_marks > user_adjusted)) + 1
            total = len(adjusted_marks)

            with result_col:
                st.markdown(f"""
                    **Your adjusted mark is:** `{user_adjusted}`  
                    **Your rank is:** `{rank}` out of `{total}` students.
                """)

    # --- Line Graph ---
    st.subheader("📈 Distribution of Marks (Original vs Adjusted)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(original_sorted)), original_sorted, marker='o', linestyle='-', label='Original', color='#FF6B6B')
    ax.plot(range(len(adjusted_sorted)), adjusted_sorted, marker='o', linestyle='-', label='Adjusted', color='#4D96FF')
    ax.set_ylim(0, 10)
    ax.set_xlabel("Student Index")
    ax.set_ylabel("Mark")
    ax.set_title("Distribution of Marks (Original vs Adjusted)")
    ax.legend()
    st.pyplot(fig)

except Exception:
    st.warning("⚠️ The quiz marks are not available yet. Please check back later.")
