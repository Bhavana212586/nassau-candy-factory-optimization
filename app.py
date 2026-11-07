
        metric_cols = st.columns(3)
        metric_cols[0].metric("Current Lead Time", f"{current_prediction:.2f} days")
        metric_cols[1].metric("Recommended Lead Time", f"{recommended_prediction:.2f} days")
        metric_cols[2].metric("Lead Time Reduction", f"{lead_reduction:.2f} days")

        st.subheader("📊 What-If Factory Comparison")
        fig = px.bar(
            scenario_df.sort_values("Predicted Lead Time"),
            x="Factory",
            y="Predicted Lead Time",
            title=f"Factory Lead Time Comparison — {product}",
            text_auto=".2f",
            color_discrete_sequence=["#142B49"]
        )
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#172B4D", family="DM Sans, sans-serif"),
            title_font_size=19,
            xaxis_title="Factory",
            yaxis_title="Predicted Lead Time (days)",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig.update_traces(
            marker_color="#142B49",
            textposition="outside",
            cliponaxis=False
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# RECOMMENDATION DASHBOARD
# =========================================================
st.markdown('<div class="section-heading">📋 Recommendation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Rank factory recommendations using the selected speed-versus-profit priority</div>', unsafe_allow_html=True)

recommendation_required = [
    "Product Name", "Current Factory", "Test Factory",
    "Lead Time Reduction", "Lead Time Reduction %", "Average Gross Profit"
]
missing_recommendation_columns = [
    column for column in recommendation_required if column not in recommendations.columns
]

if missing_recommendation_columns:
    st.warning(
        "The recommendation file is missing these columns: "
        + ", ".join(missing_recommendation_columns)
    )
else:
    filtered_recommendations = recommendations.copy()

    profit_min = filtered_recommendations["Average Gross Profit"].min()
    profit_max = filtered_recommendations["Average Gross Profit"].max()
    if pd.notna(profit_min) and pd.notna(profit_max) and profit_max != profit_min:
        filtered_recommendations["Profit Score"] = (
            (filtered_recommendations["Average Gross Profit"] - profit_min)
            / (profit_max - profit_min)
        ) * 100
    else:
        filtered_recommendations["Profit Score"] = 100.0

    reduction_min = filtered_recommendations["Lead Time Reduction %"].min()
    reduction_max = filtered_recommendations["Lead Time Reduction %"].max()
    if pd.notna(reduction_min) and pd.notna(reduction_max) and reduction_max != reduction_min:
        filtered_recommendations["Speed Score"] = (
            (filtered_recommendations["Lead Time Reduction %"] - reduction_min)
            / (reduction_max - reduction_min)
        ) * 100
    else:
        filtered_recommendations["Speed Score"] = 100.0

    speed_weight = priority / 100
    profit_weight = 1 - speed_weight
    filtered_recommendations["Priority Score"] = (
        filtered_recommendations["Speed Score"] * speed_weight
        + filtered_recommendations["Profit Score"] * profit_weight
    )

    filtered_recommendations = filtered_recommendations.sort_values(
        "Priority Score", ascending=False
    )

    display_columns = [
        "Product Name", "Current Factory", "Test Factory",
        "Lead Time Reduction", "Lead Time Reduction %",
        "Average Gross Profit", "Priority Score"
    ]
    display_recommendations = filtered_recommendations[display_columns].copy()
    for column in [
        "Lead Time Reduction", "Lead Time Reduction %",
        "Average Gross Profit", "Priority Score"
    ]:
        display_recommendations[column] = display_recommendations[column].round(2)

    st.dataframe(display_recommendations, use_container_width=True, hide_index=True)

# =========================================================
# RISK AND IMPACT PANEL
# =========================================================
st.markdown('<div class="section-heading">⚠️ Risk &amp; Impact Panel</div>', unsafe_allow_html=True)
if "Lead Time Reduction %" in recommendations.columns:
    high_impact = recommendations[recommendations["Lead Time Reduction %"] >= 10]
    st.metric("High-impact recommendations", f"{len(high_impact):,}")
else:
    st.info("Lead Time Reduction % is not available in the recommendation dataset.")

st.info(
    "Higher priority values give more importance to shipping speed. "
    "Lower priority values give more importance to profit."
)

# =========================================================
# SUCCESS MESSAGE AND FOOTER
# =========================================================
st.success(
    "The system compares factory options and recommends factory changes "
    "that may reduce predicted shipping lead time."
)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#829087;font-size:12px;'>"
    "Nassau Candy Factory Optimization · Built with Python, Pandas, Plotly &amp; Streamlit"
    "</div>",
    unsafe_allow_html=True
