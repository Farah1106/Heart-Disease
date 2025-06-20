import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.pkl")   # Trained Random Forest
scaler = joblib.load("scaler.pkl") # Scaler used in training

st.title("💓 Heart Disease Prediction App")

st.write("Enter patient data below:")

# Input fields
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 250)
fbs = st.selectbox("Fasting Blood Sugar > 120? (0=No, 1=Yes)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise-Induced Angina (0=No, 1=Yes)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope (0=Upsloping, 1=Flat, 2=Downsloping)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed, 3=Reversible)", [1, 2, 3])

# Predict button
if st.button("Predict Heart Disease"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                 restecg, thalach, exang, oldpeak, slope, ca, thal]],
                               columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    # Scale input
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is not likely to have heart disease.")
