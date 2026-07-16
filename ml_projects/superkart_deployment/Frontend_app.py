
import streamlit as st
import requests

st.title("SuperKart Sales Predictor 🚀")

# Input fields for product and store data
Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.selectbox("Product Allocated Area", ["Low", "Medium", "High"])
Product_MRP = st.number_input("Product MRP", min_value=0.0, value=142.0)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Type A", "Type B", "Type C", "Type D"])
Product_Id_char = st.selectbox("Product ID Prefix", ["FD", "NC", "DR"])
Store_Age_Years = st.number_input("Store Age (in Years)", min_value=0, value=7)
Product_Type_Category = st.selectbox("Product Type Category", ["Food", "Drinks", "Non-Consumable"])

product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

if st.button("Predict", type='primary'):
    try:
        response = requests.post(
            "https://[huggingface-username]-superkart-backend.hf.space/v1/predict",  # ✅ Correct backend URL
            json=product_data,
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            predicted_sales = result.get("Sales")
            if predicted_sales is not None:
                st.success(f"💰 Predicted Product Store Sales Total: ₹{predicted_sales:.2f}")
            else:
                st.error("❌ Unexpected response format from backend.")
        else:
            st.error(f"❌ API returned status code {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Request failed: {e}")
