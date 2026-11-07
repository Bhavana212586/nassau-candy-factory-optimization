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
# Keep all three files in the same folder as app.py.
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
# VALIDATE DATA
# =========================================================
if df.empty:
    st.error("factory_mapped_data.csv contains no records.")
    st.stop()

missing_data_columns = [
    column for column in DATA_COLUMNS if column not in df.columns
]

if missing_data_columns:
    st.error(
        "Missing columns in factory_mapped_data.csv: "
        + ", ".join(missing_data_columns)
    )
    st.stop()

missing_recommendation_columns = [
    column
    for column in RECOMMENDATION_COLUMNS
    if column not in recommendations.columns
]

if missing_recommendation_columns:
    st.error(
        "Missing columns in final_recommendations.csv: "
        + ", ".join(missing_recommendation_columns)
    )
    st.stop()

# Prepare categorical input columns consistently.
for column in DATA_COLUMNS:
    df[column] = df[column].fillna("Unknown").astype(str)

# Convert recommendation measures to numeric values.
for column in [
    "Lead Time Reduction",
    "Lead Time Reduction %",
    "Average Gross Profit"
]:
    recommendations[column] = pd.to_numeric(
        recommendations[column],
        errors="coerce"
    )

# Fit the same type of encoder used by the application.
try:
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(df[DATA_COLUMNS])
except Exception as error:
    st.error(f"Could not prepare the categorical encoder: {error}")
    st.stop()

# The saved Linear Regression model was trained with 28 one-hot encoded
# features. Check this before the user starts making predictions.
encoded_feature_count = encoder.transform(df[DATA_COLUMNS].head(1)).shape[1]
expected_feature_count = getattr(model, "n_features_in_", None)

if expected_feature_count is not None and encoded_feature_count != expected_feature_count:
    st.error(
        "The saved model and input encoder do not match. "
        f"The model expects {expected_feature_count} features, but the "
        f"current dataset produces {encoded_feature_count}. "
        "Use the matching trained model and dataset."
    )
    st.stop()

factories = sorted(df["Current Factory"].dropna().unique().tolist())
products = sorted(df["Product Name"].dropna().unique().tolist())
regions = sorted(df["Region"].dropna().unique().tolist())
ship_modes = sorted(df["Ship Mode"].dropna().unique().tolist())

if not factories or not products or not regions or not ship_modes:
    st.error(
        "The dataset must contain valid products, factories, regions, "
        "and shipping modes."
    )
    st.stop()


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
overview_cols[1].metric(
    "🍬 Total Products",
    f"{df['Product Name'].nunique():,}"
)
overview_cols[2].metric(
    "🏭 Total Factories",
    f"{df['Current Factory'].nunique():,}"
)
overview_cols[3].metric(
    "📊 Recommendations",
    f"{len(recommendations):,}"
)


# =========================================================
# SIDEBAR FILTERS
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
        "Speed ↔ Profit",
        min_value=0,
        max_value=100,
        value=70,
        help="0 = Profit priority; 100 = Shipping speed priority."
    )

    st.caption("Move the slider to change the recommendation ranking.")


# =========================================================
# FACTORY OPTIMIZATION SIMULATOR
# =========================================================
st.header("🔄 Factory Optimization Simulator")
st.write(
    "Compare predicted shipping lead times across available factories."
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
    current_factory = str(product_data["Current Factory"].iloc[0])
    scenario_results = []

    for factory in factories:
        scenario = pd.DataFrame(
            [{
                "Product Name": product,
                "Current Factory": factory,
                "Region": region,
                "Ship Mode": ship_mode
            }],
            columns=DATA_COLUMNS
        )

        try:
            scenario_encoded = encoder.transform(scenario[DATA_COLUMNS])
            prediction = float(model.predict(scenario_encoded)[0])

            scenario_results.append(
                {
                    "Factory": factory,
                    "Predicted Lead Time": prediction
                }
            )

        except Exception as error:
            st.error(
                "The trained model could not process the selected inputs. "
                "The model and encoder must use the same four input features "
                "(Product Name, Current Factory, Region, Ship Mode) and "
                "the same preprocessing used during training. "
                f"Technical details: {error}"
            )
            st.stop()

    scenario_df = pd.DataFrame(scenario_results)

    current_rows = scenario_df[
        scenario_df["Factory"] == current_factory
    ]

    if current_rows.empty:
        st.warning(
            "The selected product's current factory was not found "
            "in the factory comparison list."
        )
    else:
        current_prediction = float(
            current_rows["Predicted Lead Time"].iloc[0]
        )

        recommended_row = scenario_df.loc[
            scenario_df["Predicted Lead Time"].idxmin()
        ]

        recommended_factory = str(recommended_row["Factory"])
        recommended_prediction = float(
            recommended_row["Predicted Lead Time"]
        )

        lead_reduction = current_prediction - recommended_prediction

        st.subheader("Recommendation")

        if recommended_factory == current_factory:
            st.success(
                f"Your current factory ({current_factory}) is already "
                "the best option for this selection based on predicted "
                "lead time."
            )
        else:
            st.success(f"Recommended Factory: {recommended_factory}")

        factory_cols = st.columns(2)
        factory_cols[0].write(f"**Current Factory:** {current_factory}")
        factory_cols[1].write(
            f"**Recommended Factory:** {recommended_factory}"
        )

        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Current Lead Time",
            f"{current_prediction:.2f} days"
        )
        metric_cols[1].metric(
            "Recommended Lead Time",
            f"{recommended_prediction:.2f} days"
        )
        metric_cols[2].metric(
            "Lead Time Reduction",
            f"{lead_reduction:.2f} days"
        )

        st.subheader("📊 What-If Factory Comparison")

        chart_data = scenario_df.sort_values(
            "Predicted Lead Time",
            ascending=True
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


# =========================================================
# RECOMMENDATION DASHBOARD
# =========================================================
st.header("📋 Recommendation Dashboard")

if recommendations.empty:
    st.info("There are no recommendation records to display.")

else:
    filtered_recommendations = recommendations.copy()

    # Ensure the numeric columns contain valid numbers.
    for column in [
        "Lead Time Reduction",
        "Lead Time Reduction %",
        "Average Gross Profit"
    ]:
        filtered_recommendations[column] = pd.to_numeric(
            filtered_recommendations[column],
            errors="coerce"
        )

    # Replace missing numeric values for safe ranking.
    filtered_recommendations[
        ["Lead Time Reduction", "Lead Time Reduction %",
         "Average Gross Profit"]
    ] = filtered_recommendations[
        ["Lead Time Reduction", "Lead Time Reduction %",
         "Average Gross Profit"]
    ].fillna(0)

    # Normalize gross profit to a 0-100 score.
    profit_min = filtered_recommendations[
        "Average Gross Profit"
    ].min()

    profit_max = filtered_recommendations[
        "Average Gross Profit"
    ].max()

    if profit_max != profit_min:
        filtered_recommendations["Profit Score"] = (
            (
                filtered_recommendations["Average Gross Profit"]
                - profit_min
            ) / (profit_max - profit_min)
        ) * 100
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
