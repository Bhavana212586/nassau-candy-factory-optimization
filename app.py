import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="🏭",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("factory_mapped_data.csv")
model = joblib.load("best_shipping_model.pkl")
recommendations = pd.read_csv("final_recommendations.csv")


# Recreate the encoder used during model training
encoder = OneHotEncoder(handle_unknown="ignore")

encoder.fit(
    df[
        [
            "Product Name",
            "Current Factory",
            "Region",
            "Ship Mode"
        ]
    ]
)


factories = sorted(df["Current Factory"].dropna().unique())


# =========================================================
# TITLE
# =========================================================

# =========================================================
# PROFESSIONAL HEADER
# =========================================================

st.markdown("""
<div style="
    background: linear-gradient(120deg, #142B49, #245A91);
    padding: 28px;
    border-radius: 14px;
    margin-bottom: 25px;
">
    <h1 style="color:white; margin:0;">
        🏭 Nassau Candy Analytics
    </h1>
    <p style="color:#DCE8F5; font-size:16px; margin-top:10px;">
        Factory Reallocation & Shipping Optimization System
    </p>
    <p style="color:#DCE8F5; font-size:13px;">
        Data-driven factory allocation and shipping decisions
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# KPI SECTION
# =========================================================


# =========================================================
# PROFESSIONAL KPI SECTION
# =========================================================

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #DCE3EC;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

div[data-testid="stMetricLabel"] {
    color: #526477;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #17365D;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

col1.metric("📦 Total Orders", len(df))
col2.metric("🍬 Total Products", df["Product Name"].nunique())
col3.metric("🏭 Total Factories", df["Current Factory"].nunique())
col4.metric("📊 Recommendations", len(recommendations))


# =========================================================
# CONTROLS
# =========================================================


# =========================================================
# SIDEBAR FILTERS
# =========================================================


# =========================================================
# SIDEBAR FILTERS
# =========================================================

with st.sidebar:

    st.markdown("""
        <h2 style="color:#17365D;">
            🎛️ Dashboard Filters
        </h2>
        <p style="color:#64748B; font-size:13px;">
            Customize your optimization analysis
        </p>
    """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Factory Selection")

    product = st.selectbox(
        "Select Product",
        sorted(df["Product Name"].unique())
    )

    region = st.selectbox(
        "Select Region",
        sorted(df["Region"].dropna().unique())
    )

    ship_mode = st.selectbox(
        "Select Ship Mode",
        sorted(df["Ship Mode"].dropna().unique())
    )

    st.divider()

    st.subheader("Optimization Priority")

    priority = st.slider(
        "Speed ↔ Profit",
        min_value=0,
        max_value=100,
        value=70,
        help="0 = Profit priority, 100 = Speed priority"
    )

    st.caption("Adjust filters to explore factory optimization.")


# =========================================================
# FACTORY OPTIMIZATION SIMULATOR
# =========================================================


# =========================================================
# FACTORY OPTIMIZATION SIMULATOR
# =========================================================

st.header("🔄 Factory Optimization Simulator")

product_data = df[df["Product Name"] == product]

if not product_data.empty:

    current_factory = product_data["Current Factory"].iloc[0]

    scenario_results = []

    for factory in factories:

        scenario = pd.DataFrame({
            "Product Name": [product],
            "Current Factory": [factory],
            "Region": [region],
            "Ship Mode": [ship_mode]
        })

        scenario_encoded = encoder.transform(scenario)

        predicted_lead_time = model.predict(
            scenario_encoded
        )[0]

        scenario_results.append({
            "Factory": factory,
            "Predicted Lead Time": predicted_lead_time
        })

    scenario_df = pd.DataFrame(scenario_results)

    current_prediction = scenario_df.loc[
        scenario_df["Factory"] == current_factory,
        "Predicted Lead Time"
    ].iloc[0]

    recommended_row = scenario_df.loc[
        scenario_df["Predicted Lead Time"].idxmin()
    ]

    recommended_factory = recommended_row["Factory"]
    recommended_prediction = recommended_row["Predicted Lead Time"]

    lead_reduction = (
        current_prediction - recommended_prediction
    )


    # =====================================================
    # RECOMMENDATION
    # =====================================================

    st.subheader("Recommendation")

    col1, col2 = st.columns(2)

    col1.write(
        f"**Current Factory:** {current_factory}"
    )

    col2.write(
        f"**Recommended Factory:** {recommended_factory}"
    )


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Lead Time",
        f"{current_prediction:.2f} days"
    )

    col2.metric(
        "Recommended Lead Time",
        f"{recommended_prediction:.2f} days"
    )

    col3.metric(
        "Lead Time Reduction",
        f"{lead_reduction:.2f} days"
    )


    # =====================================================
    # WHAT-IF SCENARIO CHART
    # =====================================================

    st.subheader("📊 What-If Factory Comparison")

    fig = px.bar(
        scenario_df,
        x="Factory",
        y="Predicted Lead Time",
        title=f"Factory Lead Time Comparison - {product}",
        color="Factory",
        text_auto=".2f"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#17365D"),
        title_font_size=20,
        xaxis_title="Factory",
        yaxis_title="Predicted Lead Time (Days)",
        showlegend=False
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )



# =========================================================
# RECOMMENDATION DASHBOARD
# =========================================================

st.header("📋 Recommendation Dashboard")

filtered_recommendations = recommendations.copy()

# Calculate a simple priority score
profit_min = recommendations["Average Gross Profit"].min()
profit_max = recommendations["Average Gross Profit"].max()

if profit_max != profit_min:
    filtered_recommendations["Profit Score"] = (
        (filtered_recommendations["Average Gross Profit"] - profit_min)
        / (profit_max - profit_min)
    ) * 100
else:
    filtered_recommendations["Profit Score"] = 100

reduction_min = recommendations["Lead Time Reduction %"].min()
reduction_max = recommendations["Lead Time Reduction %"].max()

if reduction_max != reduction_min:
    filtered_recommendations["Speed Score"] = (
        (filtered_recommendations["Lead Time Reduction %"] - reduction_min)
        / (reduction_max - reduction_min)
    ) * 100
else:
    filtered_recommendations["Speed Score"] = 100


speed_weight = priority / 100
profit_weight = 1 - speed_weight

filtered_recommendations["Priority Score"] = (
    filtered_recommendations["Speed Score"] * speed_weight
    + filtered_recommendations["Profit Score"] * profit_weight
)

filtered_recommendations = filtered_recommendations.sort_values(
    "Priority Score",
    ascending=False
)


st.dataframe(
    filtered_recommendations[
        [
            "Product Name",
            "Current Factory",
            "Test Factory",
            "Lead Time Reduction",
            "Lead Time Reduction %",
            "Average Gross Profit",
            "Priority Score"
        ]
    ],
    use_container_width=True
)


# =========================================================
# RISK & IMPACT PANEL
# =========================================================

st.header("⚠️ Risk & Impact Panel")

high_impact = recommendations[
    recommendations["Lead Time Reduction %"] >= 10
]

st.write(
    "**High-impact recommendations:**",
    len(high_impact)
)

st.info(
    "Higher priority values give more importance to shipping speed. "
    "Lower priority values give more importance to profit."
)


# =========================================================
# SUCCESS MESSAGE
# =========================================================

st.success(
    "The system compares factory options and recommends "
    "factory changes that can reduce predicted shipping lead time."
)
