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
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL DARK THEME
# =========================================================
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
    color: #F5F7F5;
}

.hero {
    background: linear-gradient(120deg, #142B49, #245A91);
    border: 1px solid #376B9D;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 24px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 36px;
    margin: 0;
}

.hero p {
    color: #DCE8F5 !important;
    margin-bottom: 0;
}

.section-heading {
    color: #FFFFFF;
    font-size: 21px;
    font-weight: 700;
    margin-top: 24px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #B3C0B6;
    font-size: 13px;
    margin-bottom: 14px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #1B241F !important;
    padding: 20px;
    border-radius: 13px;
    border: 1px solid #2B3B30;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
}

div[data-testid="stMetricLabel"] {
    color: #B3C0B6 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800;
}

/* Sidebar controls */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background: #26332B !important;
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #26332B !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #FFFFFF !important;
}

[data-testid="stSlider"] {
    color: #1DB954 !important;
}

/* Alerts and recommendation messages */
[data-testid="stAlert"] {
    background: #1B3527 !important;
    border: 1px solid #2B6844 !important;
    border-radius: 10px;
}

[data-testid="stAlert"] * {
    color: #FFFFFF !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #2B3B30;
    border-radius: 12px;
}

/* Buttons */
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
# LOAD DATA AND MODEL
# =========================================================
@st.cache_data
def load_tables():
    data = pd.read_csv("factory_mapped_data.csv")
    recs = pd.read_csv("final_recommendations.csv")
    return data, recs


@st.cache_resource
def load_model():
    return joblib.load("best_shipping_model.pkl")


try:
    df, recommendations = load_tables()
    model = load_model()
except FileNotFoundError as error:
    st.error(
        "A required project file is missing. Make sure these files are in "
        "the same GitHub repository as app.py: factory_mapped_data.csv, "
        "final_recommendations.csv, and best_shipping_model.pkl."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the project data or model: {error}")
    st.stop()

required_columns = ["Product Name", "Current Factory", "Region", "Ship Mode"]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    st.error("Missing columns in factory_mapped_data.csv: " + ", ".join(missing_columns))
    st.stop()

if df.empty:
    st.error("The factory dataset is empty. Please check factory_mapped_data.csv.")
    st.stop()

# Recreate the encoder used by the original application.
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(df[required_columns])

factories = sorted(df["Current Factory"].dropna().unique().tolist())
products = sorted(df["Product Name"].dropna().unique().tolist())
regions = sorted(df["Region"].dropna().unique().tolist())
ship_modes = sorted(df["Ship Mode"].dropna().unique().tolist())

if not products or not regions or not ship_modes or not factories:
    st.error("The dataset does not contain enough valid product, region, shipping, or factory values.")
    st.stop()

# =========================================================
# PROFESSIONAL HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🏭 Nassau Candy Analytics</h1>
    <p style="font-size:16px; margin-top:10px;">
        Factory Reallocation &amp; Shipping Optimization System
    </p>
    <p style="font-size:13px; margin-top:8px;">
        Data-driven factory allocation and shipping decisions
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-heading">📌 Dashboard Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">A quick summary of the factory and recommendation dataset</div>', unsafe_allow_html=True)

overview_cols = st.columns(4)
overview_cols[0].metric("📦 Total Orders", f"{len(df):,}")
overview_cols[1].metric("🍬 Total Products", f"{df['Product Name'].nunique():,}")
overview_cols[2].metric("🏭 Total Factories", f"{df['Current Factory'].nunique():,}")
overview_cols[3].metric("📊 Recommendations", f"{len(recommendations):,}")

# =========================================================
# SIDEBAR FILTERS
# =========================================================
with st.sidebar:
    st.markdown("## 🎛️ Dashboard Filters")
    st.caption("Customize your optimization analysis")
    st.divider()

    st.subheader("Factory Selection")
    product = st.selectbox("Select Product", products)
    region = st.selectbox("Select Region", regions)
    ship_mode = st.selectbox("Select Ship Mode", ship_modes)

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
st.markdown('<div class="section-heading">🔄 Factory Optimization Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Compare predicted shipping lead times across available factories</div>', unsafe_allow_html=True)

product_data = df[df["Product Name"] == product]

if product_data.empty:
    st.warning("No records were found for the selected product.")
else:
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
        predicted_lead_time = float(model.predict(scenario_encoded)[0])
        scenario_results.append({
            "Factory": factory,
            "Predicted Lead Time": predicted_lead_time
        })

    scenario_df = pd.DataFrame(scenario_results)
    current_rows = scenario_df.loc[
        scenario_df["Factory"] == current_factory,
        "Predicted Lead Time"
    ]

    if current_rows.empty:
        st.warning("The selected product's current factory was not found in the factory list.")
    else:
        current_prediction = float(current_rows.iloc[0])
        recommended_row = scenario_df.loc[scenario_df["Predicted Lead Time"].idxmin()]
        recommended_factory = recommended_row["Factory"]
        recommended_prediction = float(recommended_row["Predicted Lead Time"])
        lead_reduction = current_prediction - recommended_prediction

        st.subheader("Recommendation")
        if recommended_factory == current_factory:
            st.success(
                f"Your current factory ({current_factory}) is already the best option for this selection."
            )
        else:
            st.success(f"Recommended Factory: {recommended_factory}")

        factory_cols = st.columns(2)
        factory_cols[0].write(f"**Current Factory:** {current_factory}")
        factory_cols[1].write(f"**Recommended Factory:** {recommended_factory}")

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
            color="Predicted Lead Time",
            color_continuous_scale=["#1DB954", "#28583A", "#8FAF9A"]
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#101411",
            paper_bgcolor="#101411",
            font=dict(color="#F5F7F5", family="DM Sans, sans-serif"),
            title_font_size=19,
            xaxis_title="Factory",
            yaxis_title="Predicted Lead Time (days)",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
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
)
