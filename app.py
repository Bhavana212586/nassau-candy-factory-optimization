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

if recommendations.empty:
    st.info("There are no recommendation records to display.")
else:
    filtered_recommendations = recommendations.copy()

    # Normalize gross profit into a 0–100 score.
    profit_min = filtered_recommendations["Average Gross Profit"].min()
    profit_max = filtered_recommendations["Average Gross Profit"].max()

    if pd.isna(profit_min) or pd.isna(profit_max):
        filtered_recommendations["Profit Score"] = 0.0
    elif profit_max != profit_min:
        filtered_recommendations["Profit Score"] = (
            (filtered_recommendations["Average Gross Profit"] - profit_min)
            / (profit_max - profit_min)
        ) * 100
    else:
        filtered_recommendations["Profit Score"] = 100.0

    # Normalize lead-time reduction percentage into a 0–100 score.
    speed_min = filtered_recommendations["Lead Time Reduction %"].min()
    speed_max = filtered_recommendations["Lead Time Reduction %"].max()

    if pd.isna(speed_min) or pd.isna(speed_max):
        filtered_recommendations["Speed Score"] = 0.0
    elif speed_max != speed_min:
        filtered_recommendations["Speed Score"] = (
            (filtered_recommendations["Lead Time Reduction %"] - speed_min)
            / (speed_max - speed_min)
        ) * 100
    else:
        filtered_recommendations["Speed Score"] = 100.0

    speed_weight = priority / 100.0
    profit_weight = 1.0 - speed_weight

    filtered_recommendations["Priority Score"] = (
        filtered_recommendations["Speed Score"] * speed_weight
        + filtered_recommendations["Profit Score"] * profit_weight
    )

    filtered_recommendations = filtered_recommendations.sort_values(
        "Priority Score", ascending=False
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
    display_data = filtered_recommendations[display_columns].copy()

    for column in [
        "Lead Time Reduction",
        "Lead Time Reduction %",
        "Average Gross Profit",
        "Priority Score"
    ]:
        display_data[column] = display_data[column].round(2)

    st.dataframe(display_data, use_container_width=True)

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
    st.write("No recommendation data is available for risk analysis.")
else:
    high_impact = recommendations[
        recommendations["Lead Time Reduction %"] >= 10
    ]
    st.metric("High-Impact Recommendations (10% or more)", len(high_impact))

st.info(
    "A higher priority value gives more importance to shipping speed. "
    "A lower priority value gives more importance to average gross profit. "
    "Factory predictions depend on the trained model and the selected inputs."
)

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption("Nassau Candy Factory Reallocation & Shipping Optimization System")

