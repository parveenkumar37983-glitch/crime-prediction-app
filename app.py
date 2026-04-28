import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ------------------ LOAD ------------------
model = pickle.load(open("model.pkl", "rb"))
le_city = pickle.load(open("le_city.pkl", "rb"))
le_weapon = pickle.load(open("le_weapon.pkl", "rb"))
le_domain = pickle.load(open("le_domain.pkl", "rb"))
le_crime = pickle.load(open("le_crime.pkl", "rb"))

# ------------------ UI ------------------
st.set_page_config(page_title="Crime Prediction", layout="centered")

st.title("🔍 Crime Prediction System")

st.markdown("Enter details to predict crime type")

# Sidebar (extra)
st.sidebar.title("About")
st.sidebar.write("ML-based Crime Prediction System")
st.sidebar.write("Model: Random Forest")

# ------------------ INPUT ------------------
city = st.selectbox("City", le_city.classes_)
hour = st.slider("Hour", 0, 23, 12)
age = st.slider("Victim Age", 1, 100, 25)
weapon = st.selectbox("Weapon Used", le_weapon.classes_)
domain = st.selectbox("Crime Domain", le_domain.classes_)

# ------------------ PREDICTION ------------------
if st.button("Predict Crime"):

    try:
        city_enc = le_city.transform([city])[0]
        weapon_enc = le_weapon.transform([weapon])[0]
        domain_enc = le_domain.transform([domain])[0]

        sample = np.array([[city_enc, hour, age, weapon_enc, domain_enc]])

        pred = model.predict(sample)
        result = le_crime.inverse_transform(pred)

        st.success(f"🚨 Predicted Crime: {result[0]}")

    except Exception as e:
        st.error("⚠️ Error in prediction")

# ------------------ CHART ------------------
st.subheader("📊 Crime Distribution by City")

try:
    df = pd.read_csv("clean_data/crime_clean.csv")
    st.bar_chart(df["City"].value_counts())
except:
    st.warning("Dataset not found for chart")

# streamlit run app.py