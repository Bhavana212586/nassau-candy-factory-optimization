import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder

    else:
        filtered_recommendations["Profit Score"] = 100.0

    # Normalize lead-time reduction percentage to a 0-100 score.
    speed_min = filtered_recommendations[
        "Lead Time Reduction %"
    ].min()

    speed_max = filtered_recommendations[
        "Lead Time Reduction %"
    ].max()

    if speed_max != speed_min:
        filtered_recommendations["Speed Score"] = (
            (
                filtered_recommendations["Lead Time Reduction %"]
                - speed_min
            ) / (speed_max - speed_min)
        ) * 100
    else:
        filtered_recommendations["Speed Score"] = 100.0

    # Calculate combined priority score.
    speed_weight = priority / 100.0
    profit_weight = 1.0 - speed_weight

    filtered_recommendations["Priority Score"] = (
        filtered_recommendations["Speed Score"] * speed_weight
        + filtered_recommendations["Profit Score"] * profit_weight
    )

    filtered_recommendations = filtered_recommendations.sort_values(
        by="Priority Score",
        ascending=False
    )

    display_columns = [
        "Product Name",
        "Current Factory",
        "Test Factory",
        "Lead Time Reduction",
        "Lead Time Reduction %",
        "Average Gross Profit",
        "Priority Score"
    ]

    display_data = filtered_recommendations[
        display_columns
    ].copy()

    for column in [
        "Lead Time Reduction",
        "Lead Time Reduction %",
        "Average Gross Profit",
        "Priority Score"
    ]:
        display_data[column] = display_data[column].round(2)

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

    csv_data = display_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Recommendations",
        data=csv_data,
        file_name="nassau_candy_recommendations.csv",
        mime="text/csv"
    )


# =========================================================
# RISK & IMPACT PANEL
# =========================================================
st.header("⚠️ Risk & Impact Panel")

if recommendations.empty:
    st.info("No recommendation data is available for risk analysis.")

else:
    reduction_values = pd.to_numeric(
        recommendations["Lead Time Reduction %"],
        errors="coerce"
    )

    high_impact = reduction_values[reduction_values >= 10]

    risk_cols = st.columns(2)
    risk_cols[0].metric(
        "High-Impact Recommendations (10% or more)",
        f"{len(high_impact):,}"
    )
    risk_cols[1].metric(
        "Total Recommendations",
        f"{len(recommendations):,}"
    )

st.info(
    "A higher priority value gives more importance to shipping speed. "
    "A lower priority value gives more importance to average gross profit. "
    "Factory predictions depend on the trained model and selected inputs."
)


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "Nassau Candy Factory Reallocation & Shipping Optimization System"
)

