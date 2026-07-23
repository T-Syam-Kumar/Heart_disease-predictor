import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient's medical details below to predict whether they have heart disease.")

# -----------------------------------------
# Load Dataset
# -----------------------------------------
heart_data = pd.read_csv("heart.csv")

# Features and Target
X = heart_data.drop(columns="target", axis=1)
Y = heart_data["target"]

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# -----------------------------------------
# Sidebar
# -----------------------------------------
st.sidebar.header("About")
st.sidebar.write(
    "This application predicts whether a person is likely to have heart disease using Logistic Regression."
)

# -----------------------------------------
# User Inputs
# -----------------------------------------

age = st.number_input("Age", 1, 120, 50)

sex = st.selectbox(
    "Sex",
    [0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type (cp)",
    [0, 1, 2, 3]
)

trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)

chol = st.number_input("Serum Cholesterol", 100, 600, 200)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG Results",
    [0, 1, 2]
)

thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels (ca)",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# -----------------------------------------
# Prediction Button
# -----------------------------------------

if st.button("Predict Heart Disease"):

    input_data = np.array([
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have Heart Disease.")
    else:
        st.success("✅ The person is not likely to have Heart Disease.")

    st.subheader("Prediction Probability")

    st.write(f"Probability of No Heart Disease: **{probability[0][0]:.2%}**")
    st.write(f"Probability of Heart Disease: **{probability[0][1]:.2%}**")

# -----------------------------------------
# Dataset Preview
# -----------------------------------------

with st.expander("View Dataset"):
    st.dataframe(heart_data)

with st.expander("Dataset Information"):
    st.write("Shape:", heart_data.shape)
    st.write("Missing Values")
    st.write(heart_data.isnull().sum())

with st.expander("Statistical Summary"):
    st.dataframe(heart_data.describe())
