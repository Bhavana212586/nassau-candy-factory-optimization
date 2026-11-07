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

# ---------------------------------------------------------
# PROFESSIONAL DARK THEME
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background: #101411 !important;
    color: #F5F7F5 !important;
}

[data-testid="stHeader"] {
    background: #101411 !important;
}

[data-testid="stSidebar"] {
    background: #151B18 !important;
    border-right: 1px solid #26332B;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.stApp h1, .stApp h2, .stApp h3,
.stApp p, .stApp label {
    color: #F5F7F5 !important;
}

.hero {
    background: linear-gradient(115deg, #1B3527, #10251B) !important;
    border: 1px solid #2B6844;
    border-radius: 22px;
    padding: 30px 34px;
    margin-bottom: 25px;
}

.hero-kicker {
    color: #9BE7B2 !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 38px;
    font-weight: 800;
}

.hero p {
    color: #D0E2D5 !important;
}

.kpi-card {
    background: #1B241F !important;
    border: 1px solid #2B3B30;
    border-radius: 16px;
    padding: 19px 20px;
}

.kpi-label {
    color: #AAB8AE !important;
}

.kpi-value {
    color: #FFFFFF !important;
    font-size: 29px;
    font-weight: 800;
}

.kpi-note {
    color: #8FD6A5 !important;
}

.section-heading {
    color: #FFFFFF !important;
    font-size: 21px;
    font-weight: 700;
    margin: 26px 0 5px;
}

.section-subtitle {
    color: #B3C0B6 !important;
    font-size: 13px;
    margin-bottom: 15px;
}

/* Sidebar input fields */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background: #26332B !important;
    color: #FFFFFF !important;
}

/* Dropdown */
[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #26332B !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* Selected dropdown tags */
[data-baseweb="tag"] {
    background: #1DB954 !important;
}

[data-baseweb="tag"] * {
    color: #07130B !important;
}

/* Slider */
[data-testid="stSlider"] {
    color: #1DB954 !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #2B3B30;
    border-radius: 12px;
}

/* Download button */
.stDownloadButton button {
    background: #1DB954 !important;
    color: #07130B !important;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}

.stDownloadButton button:hover {
    background: #35D36D !important;
}

hr {
    border-color: #29362E;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)



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
    background: #1B241F !important;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #2B3B30;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

div[data-testid="stMetricLabel"] {
    color: #B3C0B6 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800;
}

[data-testid="stAlert"] {
    background: #1B3527 !important;
    color: #FFFFFF !important;
}

[data-testid="stAlert"] * {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)

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
    if recommended_factory == current_factory:
        st.success(
            f"Your current factory ({current_factory}) "
            "is already the best option for this selection."
        )
    else:
        st.success(
            f"Recommended Factory: {recommended_factory}"
        )
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
        text_auto=".2f"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#17365D"),
        title_font_size=20,
        xaxis_title="Factory",
        yaxis_title="Predicted Lead Time",
        showlegend=False
    )

    fig.update_traces(
        marker_color="#17365D",
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

filtered_recommendations["Priority Score"] = (
    filtered_recommendations["Priority Score"].round(2)
)

filtered_recommendations["Average Gross Profit"] = (
    filtered_recommendations["Average Gross Profit"].round(2)
)

filtered_recommendations["Lead Time Reduction"] = (
    filtered_recommendations["Lead Time Reduction"].round(2)
)

filtered_recommendations["Lead Time Reduction %"] = (
    filtered_recommendations["Lead Time Reduction %"].round(2)
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
