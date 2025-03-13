import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from PIL import Image

# Load an image from a file
image = Image.open(r"C:\Users\nsiva\Downloads\shutterstock_1817852297-scaled-e1627297802233-1920x1202.jpg")

# Display the image with a caption
st.title(" Crop Production Analysis")
st.image(image)

# Load trained models and scaler
xgb_model = joblib.load(r"F:\DS\crop production analysis''\crop\xgb_model.pkl")  # Load XGBoost Model
scaler = joblib.load(r"F:\DS\crop production analysis''\crop\scaler(XGB).pkl")  # Load trained scaler

df_pivot = pd.read_excel(r"F:\DS\crop production analysis''\Transformed_data.xlsx")  # Load dataset for mapping

# Create mapping dictionaries
area_mapping = dict(zip(df_pivot["Area"], df_pivot["Area Code (M49)"]))
item_mapping = dict(zip(df_pivot["Item"], df_pivot["Item Code (CPC)"]))

# Streamlit UI
st.title(" Crop Production Prediction")
st.sidebar.image("C:/Users/nsiva/Downloads/R (1).png")
st.sidebar.header("Enter Details")

# User Inputs
area_name = st.sidebar.selectbox("Select Country", list(area_mapping.keys()))
crop_name = st.sidebar.selectbox("Select Crop", list(item_mapping.keys()))
area_harvested = st.sidebar.number_input("Area Harvested (ha)", min_value=0.0, step=0.1)
yield_kg_per_ha = st.sidebar.number_input("Yield per Area (kg/ha)", min_value=0.0, step=0.1)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2025, step=1)

# Get corresponding codes
area_code = area_mapping.get(area_name)
item_code = item_mapping.get(crop_name)

if area_code is None or item_code is None:
    st.error("Invalid Area or Crop selection.")
    st.stop()

# Create input DataFrame
input_data = pd.DataFrame([[year, area_code, item_code, area_harvested, yield_kg_per_ha]],
                          columns=["Year", "Area Code (M49)", "Item Code (CPC)", "Area_Harvested", "Yield_kg_per_ha"])

# Scale input data
input_scaled = scaler.transform(input_data)

# Predict
if st.sidebar.button("Predict Production"):
    prediction = xgb_model.predict(input_scaled)[0]
    st.success(f" Predicted Crop Production: {prediction:,.2f} tons")
    
    # Trend visualization
    years = np.arange(year - 5, year + 6)
    preds = [xgb_model.predict(scaler.transform([[y, area_code, item_code, area_harvested, yield_kg_per_ha]]))[0] for y in years]
    fig = px.line(x=years, y=preds, labels={'x': 'Year', 'y': 'Predicted Production'}, title=f'Production Trend for {crop_name} in {area_name}')
    st.plotly_chart(fig)

