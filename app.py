import streamlit as st
import pickle
import pandas as pd
import json
import numpy as np
import base64

def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("background.jpg")

with open("banglore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

with open("banglore_home_prices_columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]
    locations = data_columns[3:]

if "history" not in st.session_state:
    st.session_state.history = [] 

st.title("🏠 Bangalore Home Price Predictor")

location = st.selectbox("Select Location", sorted(locations))
sqft = st.number_input("Total Square Feet", min_value=100.0, step=10.0)
bhk = st.number_input("BHK (Bedrooms)", min_value=1, step=1)
bath = st.number_input("Bathrooms", min_value=1, step=1)

if st.button("Predict Price"):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bhk
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = round(model.predict([x])[0], 2)
    st.success(f"💰 Estimated Price: ₹ {predicted_price} Lakhs")
    
    st.session_state.history.append({
        "Location": location,
        "Sqft": sqft,
        "BHK": bhk,
        "Bath": bath,
        "Price (Lakhs)": predicted_price
    })


if st.session_state.history:
    st.header("📜 Prediction History")
    df = pd.DataFrame(st.session_state.history)
    st.table(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download History", csv, "prediction_history.csv", "text/csv")

