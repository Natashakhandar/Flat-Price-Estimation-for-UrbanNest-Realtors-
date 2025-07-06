import streamlit as st
import joblib
import numpy as np

st.title("🏢 UrbanNest Flat Price Estimator")

model = joblib.load('flat_price_model.pkl')

# Input Fields
area = st.slider("Area (in sq.ft)", 300, 5000, step=50)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5])
distance = st.slider("Distance to Metro (km)", 0.1, 10.0, step=0.1)
age = st.slider("Age of Flat (years)", 0, 50)
amenity = st.slider("Amenities Score", 0, 10)

if st.button("Estimate Price"):
    input_data = np.array([[area, bedrooms, distance, age, amenity]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹{prediction[0]:,.2f}")
