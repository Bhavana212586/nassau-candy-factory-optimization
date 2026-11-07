
import streamlit as st
import pandas as pd

# =========================================================
# FACTORY LEAD TIME CHART
# =========================================================

# Make sure the chart figure (fig) has already been created.
if "fig" in locals() and fig is not None:
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#17365D"),
        title_font=dict(color="#17365D", size=19),
        xaxis_title="Factory",
        yaxis_title="Predicted Lead Time (days)",
        showlegend=False
    )

    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# RECOMMENDATION DASHBOARD
# =========================================================

st.header("📋 Recommendation Dashboard")

required_columns = [
    "Product Name",
    "Current Factory",
    "Test Factory",
    "Lead Time Reduction",
    "Lead Time Reduction %",
    "Average Gross Profit"
]

if recommendations.empty:
    st.info("There are no recommendation records to display.")

elif not all(col in recommendations.columns for col in required_columns):
    st.error("Recommendation data is missing required columns.")

else:
    filtered_recommendations = recommendations.copy()

    # Convert numeric columns safely
    numeric_columns = [
        "Lead Time Reduction",
        "Lead Time Reduction %",
        "Average Gross Profit"
    ]

    for column in numeric_columns:
        filtered_recommendations[column] = pd.to_numeric(
            filtered_recommendations[column],
            errors="coerce"
        )

    # Handle missing values
    filtered_recommendations[numeric_columns] = (
        filtered_recommendations[numeric_columns].fillna(0)
    )

    # Normalize gross profit
    profit_min = filtered_recommendations[
        "Average Gross Profit"
    ].min()

    profit_max = filtered_recommendations[
        "Average Gross Profit"
    ].max()

    if profit_max != profit_min:
        filtered_recommendations["Profit Score"] = (
            (filtered_recommendations["Average Gross Profit"] - profit_min)
            / (profit_max - profit_min)
        ) * 100
    else:
        filtered_recommendations["Profit Score"] = 100.0

    # Normalize lead time reduction
    speed_min = filtered_recommendations[
        "Lead Time Reduction %"
    ].min()

    speed_max = filtered_recommendations[
        "Lead Time Reduction %"
    ].max()

    if speed_max != speed_min:
        filtered_recommendations["Speed Score"] = (
            (filtered_recommendations["Lead Time Reduction %"] - speed_min)
            / (speed_max - speed_min)
        ) * 100
    else:
        filtered_recommendations["Speed Score"] = 100.0

    # Priority calculation
    priority_value = float(priority)
    priority_value = max(0, min(100, priority_value))

    speed_weight = priority_value / 100
    profit_weight = 1 - speed_weight

    filtered_recommendations["Priority Score"] = (
        filtered_recommendations["Speed Score"] * speed_weight
        + filtered_recommendations["Profit Score"] * profit_weight
    )

    # Sort recommendations
    filtered_recommendations = filtered_recommendations.sort_values(
        by="Priority Score",
        ascending=False
    )

    # Display columns
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

    # Round numeric values
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

    # Download CSV
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

elif "Lead Time Reduction %" not in recommendations.columns:
    st.warning("Lead Time Reduction % column is missing.")

else:
    reduction_values = pd.to_numeric(
        recommendations["Lead Time Reduction %"],
        errors="coerce"
    )

    high_impact = reduction_values[reduction_values >= 10]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "High-Impact Recommendations",
            len(high_impact)
        )

    with col2:
        st.metric(
            "Total Recommendations",
            len(recommendations)
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
