
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path


# Load the trained model
@st.cache_resource
def load_model():
    model_path = "/content/drive/My Drive/Colab Notebooks/Model Deployment/backend_files/rf_tuned.joblib"
    return joblib.load(model_path)

model = load_model()


# Streamlit UI for Sales Prediction
st.title("🛒 SuperKart Sales Forecasting App")
st.write(
    "This tool predicts product sales based on product and store details."
)

st.subheader("Enter the product and store details:")

# Collect user input
product_weight = st.number_input(
    "Product Weight", min_value=0.0, value=12.0, step=0.1
)

product_sugar_content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

product_allocated_area = st.number_input(
    "Product Allocated Area", min_value=0.0, value=0.05, step=0.01
)

product_mrp = st.number_input(
    "Product MRP", min_value=0.0, value=150.0, step=1.0
)

store_size = st.selectbox(
    "Store Size", ["Small", "Medium", "High"]
)

store_location_city_type = st.selectbox(
    "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
)

store_type = st.selectbox(
    "Store Type",
    [
        "Grocery Store",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3"
    ]
)

product_id_char = st.selectbox(
    "Product ID Character", ["FD", "DR", "NC"]
)

store_age_years = st.number_input(
    "Store Age (Years)", min_value=0, value=10, step=1
)

product_type_category = st.selectbox(
    "Product Type Category", ["Food", "Drinks", "Non-Consumable"]
)

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_MRP": product_mrp,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location_city_type,
    "Store_Type": store_type,
    "Product_Id_char": product_id_char,
    "Store_Age_Years": store_age_years,
    "Product_Type_Category": product_type_category
}])

# Predict button
if st.button("Predict Sales"):
    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Sales Prediction")
        st.success(f"Predicted Sales: {prediction:,.2f}")

    except Exception as e:
        st.error(f"Unable to generate prediction: {e}")
