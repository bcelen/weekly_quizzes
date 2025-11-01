import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Z-Score Transformer", layout="centered")
st.title("\U0001F4C8 MBS Grade Adjustment App")

uploaded_file = st.file_uploader("\U0001F4C4 Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, header=None, skiprows=8)

    if len(df_raw.columns) <= 10:
        st.error("❌ Column K not found.")
    else:
        sample_col = df_raw.iloc[:, 10]
        st.write("📄 Raw Column K Data", sample_col)

        numeric_sample = pd.to_numeric(sample_col, errors='coerce')
        valid_mask = ~numeric_sample.isna()
        sample_numeric = numeric_sample[valid_mask].values

        if len(sample_numeric) == 0:
            st.error("❌ Column K has no numeric values.")
            st.stop()

        # 🎯 User inputs
        new_mean = st.slider("\U0001F3AF New Mean", min_value=74.0, max_value=76.0, value=75.0, step=0.1)
        target_pct_above_80 = st.slider("\U0001F3AF % of values above 80", 0.20, 0.30, 0.25)

        # 🔢 Z-score transformation
        mean_orig = np.mean(sample_numeric)
        std_orig = np.std(sample_numeric)
        z_scores = (sample_numeric - mean_orig) / std_orig

        z_target = norm.ppf(1 - target_pct_above_80)
        required_std = (80 - new_mean) / z_target
        adjusted_values = np.clip(z_scores * required_std + new_mean, 0, 100)

        # 🧱 Reconstruct full output
        zscore_full = pd.Series([None] * len(sample_col))
        new_sample_full = pd.Series([None] * len(sample_col))
        zscore_full[valid_mask] = z_scores
        new_sample_full[valid_mask] = adjusted_values
        new_sample_full[~valid_mask] = sample_col[~valid_mask]

        # 📄 Final output DataFrame with rounding
        df_out = pd.DataFrame({
            "Original Grades": pd.to_numeric(sample_col, errors="coerce").round(2).combine_first(sample_col),
            "Z-score": pd.to_numeric(zscore_full, errors="coerce").round(2),
            "Adjusted Grades": pd.to_numeric(new_sample_full, errors="coerce").round(2).combine_first(new_sample_full)
        })

        st.write("✅ Transformed Grades (Rounded to 2 Decimal Places)")
        st.dataframe(df_out)

        # 💾 Download Excel
        buffer = io.BytesIO()
        df_out.to_excel(buffer, index=False, engine='openpyxl')
        st.download_button(
            label="\U0001F4E5 Download Transformed Excel File",
            data=buffer.getvalue(),
            file_name="transformed_sample.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 📊 Grade Histogram
        adjusted_numeric = pd.to_numeric(new_sample_full, errors='coerce').dropna()
        grade_bins = [0, 50, 65, 70, 75, 80, 100]
        grade_labels = ['F', 'P', 'H3', 'H2B', 'H2A', 'H1']

        # Color map:
        colors = {
            'F': '#d9eaf5',
            'P': '#c0d8ec',
            'H3': '#a5c5e4',
            'H2B': '#7fb2db',
            'H2A': '#1f77b4',     # Deep blue
            'H1': '#ff7f0e'       # Orange
        }

        grade_series = pd.cut(adjusted_numeric, bins=grade_bins, labels=grade_labels, right=False)
        grade_counts = grade_series.value_counts().reindex(grade_labels, fill_value=0)
        grade_percents = (grade_counts / len(adjusted_numeric) * 100).round(2)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(grade_labels, grade_counts, color=[colors[g] for g in grade_labels], edgecolor='black')
        ax.set_xlabel("Grade")
        ax.set_ylabel("Number of Students")
        ax.set_title("\U0001F4CA Distribution of Adjusted Marks")
        st.pyplot(fig)

        # 📈 Stats
        mean_adj = adjusted_numeric.mean()
        pct_H1 = grade_percents['H1']

        st.write(f"**\U0001F4C8 Mean of Adjusted Marks:** {mean_adj:.2f}")
        st.write(f"**🔥 Percentage of H1s:** {pct_H1:.2f}%")

        # 📋 Summary Table
        summary_df = pd.DataFrame({
            "Grade": grade_labels,
            "Number": grade_counts.values,
            "% of Class": grade_percents.values
        })

        # ✅ Corrected Totals
        non_numeric = sample_col.apply(lambda x: not pd.api.types.is_number(x)).sum()
        total_results = len(sample_numeric)
        total_students = total_results + non_numeric

        st.markdown("### \U0001F4CB Summary of Overall Results")
        st.dataframe(summary_df)

        st.markdown(f"""
        **Total Results:** {total_results}  
        **Non-numeric:** {non_numeric}  
        **Students:** {total_students}  
        **Mean:** {mean_adj:.2f}  
        **Standard Deviation:** {adjusted_numeric.std():.2f}  
        """)
