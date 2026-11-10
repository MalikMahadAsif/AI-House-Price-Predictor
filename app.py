
import streamlit as st
import pandas as pd
import joblib

# Configure the web page
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load the trained machine learning model
@st.cache_resource
def load_model():
    return joblib.load("models/house_price_model.pkl")

model = load_model()

# Website heading
st.title("🏠 AI House Price Predictor")

st.write(
    "Enter California housing district details "
    "to estimate the median housing value."
)

st.info("This is an educational ML project, not a property appraisal.")

# Get input values from the user
med_inc = st.number_input("Median Income (in $10,000 units)", 0.0, 20.0, 5.0)
house_age = st.number_input("Median House Age", 1.0, 100.0, 25.0)
ave_rooms = st.number_input("Average Rooms per Household", 0.1, 50.0, 6.0)
ave_bedrms = st.number_input("Average Bedrooms per Household", 0.1, 20.0, 1.2)
population = st.number_input("District Population", 1.0, 50000.0, 1500.0)
ave_occup = st.number_input("Average Household Occupancy", 0.1, 100.0, 3.0)
latitude = st.number_input("Latitude", 32.0, 42.0, 37.0)
longitude = st.number_input("Longitude", -125.0, -114.0, -122.0)

# Predict when button is clicked
if st.button("Predict Housing Value", type="primary"):

    # Preserve the same column names and order as training
    input_data = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Convert target units to USD
    estimated_value = prediction * 100000

    st.success(f"Estimated Median Housing Value: ${estimated_value:,.2f}")
