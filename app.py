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
# PAGE STYLE
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #17365D;
    }
    [data-testid="stHeader"] {
        background-color: #FFFFFF;
    }
    [data-testid="stSidebar"] {
        background-color: #F1F4F8;
    }
    [data-testid="stSidebar"] * {
        color: #17365D;
    }
    h1, h2, h3, p, label {
        color: #17365D;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE3EC;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(23, 54, 93, 0.06);
    }
    div[data-testid="stMetricLabel"] {
        color: #526477;
    }
    div[data-testid="stMetricValue"] {
        color: #17365D;
        font-weight: 750;
    }
    .stButton button, .stDownloadButton button {
        background-color: #17365D;
        color: #FFFFFF;
        border: 0;
        border-radius: 8px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #DCE3EC;
        border-radius: 10px;
    }
    hr {
        border-color: #DCE3EC;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# FILE PATHS
# Keep these files in the same folder as app.py.
# =========================================================
DATA_FILE = "factory_mapped_data.csv"
RECOMMENDATION_FILE = "final_recommendations.csv"
MODEL_FILE = "best_shipping_model.pkl"

DATA_COLUMNS = [
    "Product Name",
    "Current Factory",
    "Region",
    "Ship Mode"
]

RECOMMENDATION_COLUMNS = [
    "Product Name",
    "Current Factory",
    "Test Factory",
    "Lead Time Reduction",
    "Lead Time Reduction %",
    "Average Gross Profit"
]

FACTORY_COLUMN = "Current Factory"


# =========================================================
# LOAD DATA AND MODEL
# =========================================================
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    recommendation_data = pd.read_csv(RECOMMENDATION_FILE)
    return data, recommendation_data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    df, recommendations = load_data()
    model = load_model()
except FileNotFoundError as error:
    missing_file = getattr(error, "filename", None) or "a required project file"
    st.error(
        f"Could not find {missing_file}. Make sure these files are in "
        "the same GitHub repository folder as app.py: "
        "factory_mapped_data.csv, final_recommendations.csv, "
        "and best_shipping_model.pkl."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the project files: {error}")
    st.stop()


# =========================================================
# VALIDATE AND PREPARE DATA
# =========================================================
if df.empty:
    st.error("factory_mapped_data.csv contains no records.")
    st.stop()

missing_data_columns = [c for c in DATA_COLUMNS if c not in df.columns]
if missing_data_columns:
    st.error(
        "Missing columns in factory_mapped_data.csv: "
        + ", ".join(missing_data_columns)
    )
    st.stop()

missing_recommendation_columns = [
    c for c in RECOMMENDATION_COLUMNS if c not in recommendations.columns
]
if missing_recommendation_columns:
    st.error(
        "Missing columns in final_recommendations.csv: "
        + ", ".join(missing_recommendation_columns)
    )
    st.stop()

for column in DATA_COLUMNS:
    df[column] = df[column].fillna("Unknown").astype(str)

numeric_columns = [
    "Lead Time Reduction",
    "Lead Time Reduction %",
    "Average Gross Profit"
]
for column in numeric_columns:
    recommendations[column] = pd.to_numeric(
        recommendations[column], errors="coerce"
    )

# Factory-level historical gross profit is used only as a proxy when
# comparing factories in the simulator. The dataset does not contain
# a separate predicted profit for every product-factory scenario.
if "Gross Profit" in df.columns:
    df["Gross Profit"] = pd.to_numeric(df["Gross Profit"], errors="coerce")
    factory_profit = (
        df.groupby(FACTORY_COLUMN)["Gross Profit"]
        .mean()
        .to_dict()
    )
else:
    factory_profit = {}

factories = sorted(df[FACTORY_COLUMN].dropna().unique().tolist())
products = sorted(df["Product Name"].dropna().unique().tolist())
regions = sorted(df["Region"].dropna().unique().tolist())
ship_modes = sorted(df["Ship Mode"].dropna().unique().tolist())

if not factories or not products or not regions or not ship_modes:
    st.error(
        "The dataset must contain valid products, factories, regions, "
        "and shipping modes."
    )
    st.stop()

# Fit encoder using the same four categorical fields used in this app.
try:
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(df[DATA_COLUMNS])
except Exception as error:
    st.error(f"Could not prepare the categorical encoder: {error}")
    st.stop()

encoded_feature_count = encoder.transform(df[DATA_COLUMNS].head(1)).shape[1]
expected_feature_count = getattr(model, "n_features_in_", None)

if (
    expected_feature_count is not None
    and encoded_feature_count != expected_feature_count
):
    st.error(
        "The saved model and input encoder do not match. "
        f"The model expects {expected_feature_count} features, but the "
        f"current dataset produces {encoded_feature_count}. "
        "Use the matching trained model and dataset."
    )
    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def min_max_score(series, higher_is_better=True):
    """Convert a numeric series to a 0-100 score."""
    values = pd.to_numeric(series, errors="coerce")
    low = values.min()
    high = values.max()

    if pd.isna(low) or pd.isna(high):
        return pd.Series(50.0, index=series.index)

    if high == low:
        return pd.Series(50.0, index=series.index)

    score = ((values - low) / (high - low)) * 100
    if not higher_is_better:
        score = 100 - score
    return score.fillna(50.0)


def add_priority_score(data, speed_priority):
    """
    0 means profit priority, 50 means balanced, 100 means speed priority.
    Speed is represented by lead-time reduction percentage.
    Profit is represented by average gross profit.
    """
    result = data.copy()

    result["Speed Score"] = min_max_score(
        result["Lead Time Reduction %"], higher_is_better=True
    )
    result["Profit Score"] = min_max_score(
        result["Average Gross Profit"], higher_is_better=True
    )

    speed_weight = speed_priority / 100.0
    profit_weight = 1.0 - speed_weight

    result["Priority Score"] = (
        result["Speed Score"] * speed_weight
        + result["Profit Score"] * profit_weight
    )
    return result


def predict_factory_lead_time(product_name, factory_name, selected_region, selected_ship_mode):
    """Predict lead time for one product/factory/region/shipping scenario."""
    scenario = pd.DataFrame(
        [{
            "Product Name": str(product_name),
            "Current Factory": str(factory_name),
            "Region": str(selected_region),
            "Ship Mode": str(selected_ship_mode)
        }],
        columns=DATA_COLUMNS
    )
    encoded = encoder.transform(scenario[DATA_COLUMNS])
    return float(model.predict(encoded)[0])


def score_factory_scenarios(scenarios, speed_priority):
    """
    Rank candidate factories using predicted speed and historical
    factory-level average gross profit. Profit is a proxy, not a
    product-specific counterfactual profit prediction.
    """
    result = scenarios.copy()

    result["Speed Score"] = min_max_score(
        result["Predicted Lead Time"], higher_is_better=False
    )
    result["Profit Score"] = min_max_score(
        result["Historical Avg Gross Profit"], higher_is_better=True
    )

    speed_weight = speed_priority / 100.0
    profit_weight = 1.0 - speed_weight

    result["Priority Score"] = (
        result["Speed Score"] * speed_weight
        + result["Profit Score"] * profit_weight
    )
    return result


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div style="
        background: linear-gradient(120deg, #142B49, #245A91);
        padding: 28px;
        border-radius: 14px;
        margin-bottom: 22px;
    ">
        <h1 style="color: white; margin: 0;">
            🏭 Nassau Candy Analytics
        </h1>
        <p style="color: #E5EDF7; font-size: 16px; margin: 8px 0 0;">
            Factory Reallocation &amp; Shipping Optimization System
        </p>
        <p style="color: #D2DEED; font-size: 13px; margin: 6px 0 0;">
            Data-driven factory allocation and shipping decisions
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD OVERVIEW
# =========================================================
st.header("📌 Dashboard Overview")

overview_cols = st.columns(4)
overview_cols[0].metric("📦 Total Orders", f"{len(df):,}")
overview_cols[1].metric("🍬 Total Products", f"{df['Product Name'].nunique():,}")
overview_cols[2].metric("🏭 Total Factories", f"{df[FACTORY_COLUMN].nunique():,}")
overview_cols[3].metric("📊 Recommendations", f"{len(recommendations):,}")


# =========================================================
# SIDEBAR FILTERS AND OPTIMIZATION PRIORITY
# =========================================================
with st.sidebar:
    st.header("🎛️ Dashboard Filters")
    st.caption("Choose the details for your factory comparison.")
    st.divider()

    product = st.selectbox("Select Product", products)
    region = st.selectbox("Select Region", regions)
    ship_mode = st.selectbox("Select Ship Mode", ship_modes)

    st.divider()
    st.subheader("Optimization Priority")

    priority = st.slider(
        "Profit  ↔  Speed",
        min_value=0,
        max_value=100,
        value=50,
        help="0 = Profit priority; 50 = Balanced; 100 = Shipping speed priority."
    )

    if priority < 40:
        st.caption("Current setting: More importance to profit")
    elif priority > 60:
        st.caption("Current setting: More importance to shipping speed")
    else:
        st.caption("Current setting: Balanced profit and shipping speed")


# =========================================================
# FACTORY OPTIMIZATION SIMULATOR
# =========================================================
st.header("🔄 Factory Optimization Simulator")
st.write(
    "Compare predicted shipping lead times and a historical profit proxy "
    "across available factories."
)

product_data = df[df["Product Name"] == product]

scenario_df = pd.DataFrame()
current_factory = None
recommended_factory = None
current_prediction = None
recommended_prediction = None
lead_reduction = None

if product_data.empty:
    st.warning("No records were found for the selected product.")
else:
    current_factory = str(product_data[FACTORY_COLUMN].mode().iloc[0])
    scenario_results = []

    for factory in factories:
        try:
            prediction = predict_factory_lead_time(
                product, factory, region, ship_mode
            )
            scenario_results.append({
                "Factory": factory,
                "Predicted Lead Time": prediction,
                "Historical Avg Gross Profit": float(
                    factory_profit.get(factory, 0.0)
                )
            })
        except Exception as error:
            st.error(
                "The trained model could not process the selected inputs. "
                "Check that the model and encoder use the same four input "
                "features and preprocessing. "
                f"Technical details: {error}"
            )
            st.stop()

    scenario_df = pd.DataFrame(scenario_results)

    current_rows = scenario_df[scenario_df["Factory"] == current_factory]
    if current_rows.empty:
        st.warning(
            "The selected product's current factory was not found "
            "in the factory comparison list."
        )
    else:
        current_prediction = float(
            current_rows["Predicted Lead Time"].iloc[0]
        )

        # The selected slider now determines the factory ranking.
        ranked_scenarios = score_factory_scenarios(scenario_df, priority)
        recommended_row = ranked_scenarios.loc[
            ranked_scenarios["Priority Score"].idxmax()
        ]

        recommended_factory = str(recommended_row["Factory"])
        recommended_prediction = float(
            recommended_row["Predicted Lead Time"]
        )
        lead_reduction = current_prediction - recommended_prediction

        st.subheader("Recommendation")

        if recommended_factory == current_factory:
            st.success(
                f"Your current factory ({current_factory}) is the best "
                "choice for the selected priority setting."
            )
        else:
            st.success(
                f"Recommended Factory: {recommended_factory}"
            )

        factory_cols = st.columns(2)
        factory_cols[0].write(f"**Current Factory:** {current_factory}")
        factory_cols[1].write(f"**Recommended Factory:** {recommended_factory}")

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Current Lead Time", f"{current_prediction:.2f} days"
        )
        metric_cols[1].metric(
            "Recommended Lead Time", f"{recommended_prediction:.2f} days"
        )
        metric_cols[2].metric(
            "Lead Time Reduction", f"{lead_reduction:.2f} days"
        )

        if lead_reduction > 0:
            st.caption(
                f"The recommended scenario is predicted to save "
                f"{lead_reduction:.2f} days."
            )
        elif lead_reduction < 0:
            st.caption(
                f"The recommended scenario is predicted to take "
                f"{abs(lead_reduction):.2f} more days. It may have a better "
                "profit score under the selected balance."
            )
        else:
            st.caption(
                "The current and recommended scenarios have the same "
                "predicted lead time."
            )

        st.subheader("📊 What-If Factory Comparison")

        chart_data = scenario_df.sort_values(
            "Predicted Lead Time", ascending=True
        )

        fig = px.bar(
            chart_data,
            x="Factory",
            y="Predicted Lead Time",
            title=f"Factory Lead Time Comparison — {product}",
            text="Predicted Lead Time",
            color_discrete_sequence=["#17365D"]
        )
        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            cliponaxis=False
        )
        fig.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#17365D"),
            title_font=dict(color="#17365D", size=19),
            xaxis_title="Factory",
            yaxis_title="Predicted Lead Time (days)",
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Profit comparison note: the dataset contains historical gross "
            "profit by existing factory assignment, not a separate predicted "
            "profit for every product-factory alternative. Therefore, the "
            "simulator uses each factory's overall historical average gross "
            "profit as a proxy. Treat this as an indicative comparison, not "
            "a guaranteed profit forecast."
        )

        with st.expander("View factory scores used for this recommendation"):
            score_view = ranked_scenarios[
                [
                    "Factory",
                    "Predicted Lead Time",
                    "Historical Avg Gross Profit",
                    "Speed Score",
                    "Profit Score",
                    "Priority Score"
                ]
            ].sort_values("Priority Score", ascending=False).copy()
            for col in [
                "Predicted Lead Time",
                "Historical Avg Gross Profit",
                "Speed Score",
                "Profit Score",
                "Priority Score"
            ]:
                score_view[col] = score_view[col].round(2)
            st.dataframe(score_view, use_container_width=True, hide_index=True)


# =========================================================
# RECOMMENDATION DASHBOARD
# =========================================================
st.header("📋 Recommendation Dashboard")

if recommendations.empty:
    st.info("There are no recommendation records to display.")
else:
    filtered_recommendations = recommendations.copy()

    for column in numeric_columns:
        filtered_recommendations[column] = pd.to_numeric(
            filtered_recommendations[column], errors="coerce"
        )

    filtered_recommendations[numeric_columns] = (
        filtered_recommendations[numeric_columns].fillna(0)
    )

    # Apply the same live priority setting to recommendation ranking.
    filtered_recommendations = add_priority_score(
        filtered_recommendations, priority
    )
    filtered_recommendations = filtered_recommendations.sort_values(
        by="Priority Score", ascending=False
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

    st.caption(
        f"Recommendations are ranked using the current slider setting: "
        f"{priority}% speed weight and {100 - priority}% profit weight."
    )
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
        recommendations["Lead Time Reduction %"], errors="coerce"
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
    "At 50, both receive equal weight. Factory predictions depend on the "
    "trained model and selected inputs."
)


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Nassau Candy Factory Reallocation & Shipping Optimization System"
)
