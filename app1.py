import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image

# Load an image from a file
image = Image.open("C:/Users/nsiva/Downloads/shutterstock_1817852297-scaled-e1627297802233-1920x1202.jpg")

# Display the image with a caption
st.title(" Crop Production Prediction Analysis")
st.image(image)
# Load transformed dataset
transformed_data = pd.read_excel("F:/DS/crop production analysis''/Transformed_data.xlsx")  # Ensure the correct path

# Load trained models and scaler
xgb_model = joblib.load("F:/DS/crop production analysis''/crop/xgb_model.pkl")  # XGBoost Model
gb_model = joblib.load("F:/DS/crop production analysis''/crop/gb_model.pkl")  # Gradient Boosting Model
rf_model = joblib.load("F:/DS/crop production analysis''/crop/rf_model.pkl")  # Random Forest Model
scaler = joblib.load("F:/DS/crop production analysis''/crop/scaler.pkl")  # Scaler
accuracy_metrics = joblib.load("F:/DS/crop production analysis''/crop/accuracy_metrics.pkl")  # Accuracy metrics

# Create mapping dictionaries
area_mapping = dict(zip(transformed_data["Area"], transformed_data["Area Code (M49)"]))
item_mapping = dict(zip(transformed_data["Item"], transformed_data["Item Code (CPC)"]))

# Streamlit UI
st.title("Crop Production Prediction")
st.sidebar.image("C:/Users/nsiva/Downloads/R (1).png")
st.sidebar.header("Enter Details")

# User Inputs
area_name = st.sidebar.selectbox("Select Country", list(area_mapping.keys()))
crop_name = st.sidebar.selectbox("Select Crop", list(item_mapping.keys()))
year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, value=2025, step=1)
area_harvested = st.sidebar.number_input("Area Harvested (ha)", min_value=0.0, step=0.1)
yield_kg_per_ha = st.sidebar.number_input("Yield per Area (kg/ha)", min_value=0.0, step=0.1)

# Select model
model_choice = st.sidebar.selectbox("Select Regression Model", ["XGBoost", "Gradient Boosting", "Random Forest"])
model_mapping = {"XGBoost": xgb_model, "Gradient Boosting": gb_model, "Random Forest": rf_model}
selected_model = model_mapping[model_choice]

# Get corresponding codes
area_code = area_mapping.get(area_name, None)
item_code = item_mapping.get(crop_name, None)

if st.sidebar.button("Predict Production"):
    if area_code is None or item_code is None:
        st.error("Invalid Area or Crop selection.")
    else:
        # Create input DataFrame
        input_data = pd.DataFrame([[year, area_code, item_code, area_harvested, yield_kg_per_ha]],
                                  columns=["Year", "Area Code (M49)", "Item Code (CPC)", "Area_Harvested", "Yield_kg_per_ha"])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Predict using selected model
        prediction = selected_model.predict(input_scaled)[0]
        
        # Display prediction
        st.write("### Predicted Crop Production")
        st.success(f"{model_choice} Regression: {prediction:.2f} tons")
        
        # Display model accuracy
        st.write("### Model Accuracy")
        metrics = accuracy_metrics.get(model_choice, {})
        st.write(f"R² Score: {metrics.get('R2', 'N/A')}")
        st.write(f"MAE: {metrics.get('MAE', 'N/A')}")
        st.write(f"MSE: {metrics.get('MSE', 'N/A')}")
        
        # Generate a dynamic plot
        years = np.arange(year - 5, year + 6)
        fig = px.line(title=f'Predicted Production Trend for {crop_name} in {area_name}')
        preds = [selected_model.predict(scaler.transform([[y, area_code, item_code, area_harvested, yield_kg_per_ha]]))[0] for y in years]
        fig.add_scatter(x=years, y=preds, mode='lines', name=f'{model_choice} Prediction')
        
        st.plotly_chart(fig)
