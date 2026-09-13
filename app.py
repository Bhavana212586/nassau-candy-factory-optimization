
import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="🏭",
    layout="wide"
)

# Load project files
df = pd.read_csv("factory_mapped_data.csv")
model = joblib.load("best_shipping_model.pkl")
recommendations = pd.read_csv("final_recommendations.csv")

# Title
st.title("🏭 Nassau Candy Factory Optimization")
st.write("Factory Reallocation & Shipping Optimization Recommendation System")

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Orders",
    len(df)
)

col2.metric(
    "Total Products",
    df["Product Name"].nunique()
)

col3.metric(
    "Total Factories",
    df["Current Factory"].nunique()
)

col4.metric(
    "Recommendations",
    len(recommendations)
)

# Factory Optimization Simulator
st.header("🔄 Factory Optimization Simulator")

product = st.selectbox(
    "Select Product",
    sorted(df["Product Name"].unique())
)

product_info = recommendations[
    recommendations["Product Name"] == product
]

if not product_info.empty:

    row = product_info.iloc[0]

    st.write("### Recommendation")

    st.write("**Current Factory:**", row["Current Factory"])
    st.write("**Recommended Factory:**", row["Test Factory"])

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Lead Time",
        f'{row["Current Predicted Lead Time"]:.2f} days'
    )

    col2.metric(
        "Recommended Lead Time",
        f'{row["Predicted Lead Time"]:.2f} days'
    )

    col3.metric(
        "Lead Time Reduction",
        f'{row["Lead Time Reduction"]:.2f} days'
    )

# Recommendation Dashboard
st.header("📊 Recommendation Dashboard")

st.dataframe(
    recommendations[
        [
            "Product Name",
            "Current Factory",
            "Test Factory",
            "Lead Time Reduction",
            "Lead Time Reduction %",
            "Average Gross Profit"
        ]
    ],
    use_container_width=True
)

# Risk and Impact
st.header("⚠️ Risk & Impact Panel")

high_impact = recommendations[
    recommendations["Lead Time Reduction %"] >= 10
]

st.write(
    "High-impact recommendations:",
    len(high_impact)
)

st.success(
    "The system identifies factory changes that can reduce predicted shipping lead time."
)
