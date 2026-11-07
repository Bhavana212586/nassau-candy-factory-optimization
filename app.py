import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="🏭",
    layout="wide"
)
# =========================================================
# PROFESSIONAL DASHBOARD HEADER
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #17365D;
    text-align: center;
    padding: 15px;
}

.sub-title {
    font-size: 17px;
    color: #526477;
    text-align: center;
    margin-bottom: 30px;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #DCE3EC;
    padding: 18px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🏭 Nassau Candy Factory Optimization</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Factory Reallocation & Shipping Optimization Recommendation System</div>',
    unsafe_allow_html=True
)

st.divider()
