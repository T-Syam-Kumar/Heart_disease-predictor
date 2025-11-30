import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------------- Page Config -----------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide",
    page_icon="❤️"
)

# ---------------------- Custom CSS ------------------------
st.markdown(
    """
    <style>
        .main {
            background-color: #f7f7f7;
        }
        .title {
            font-size: 40px;
            text-align: center;
            color: #d62828;
            font-weight: bold;
            padding-bottom: 10px;
        }
        .subtitle {
            font-size: 18px;
            text-align: center;
            color: #555;
            padding-bottom: 20px;
        }
        .box {
            background-color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .predict-btn button {
            width: 100%;
            background-color: #d62828 !important;
            color: white !important;
            height: 50px;
            font-size: 20px !important;
            border-radius: 10px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- Title -----------------------------
st.markdown("<div class='title'>❤️ Heart Disease Prediction App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter patient medical details and predict the likelihood of heart disease.</div>", unsafe_allow_html=True)

# ---------------------- Load Dataset -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")  # LOCAL dataset
    return df

df = load_data()

# ---------------------- Train Model ------------------------
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------------- Sidebar ---------------------------
st.sidebar.header("📊 Model Details")
st.sidebar.write(f"**Model:** Logistic Regression")
st.sidebar.write(f"**Accuracy:** `{accuracy:.2f}`")

if st.sidebar.checkbox("Show Dataset"):
    st.sidebar.write(df)

# ---------------------- Form UI ---------------------------
st.header("🧍 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 20, 100, 40)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 240)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", ["True", "False"])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

with col3:
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [1, 2, 3])

# Convert UI values → numerical
sex_val = 1 if sex == "Male" else 0
fbs_val = 1 if fbs == "True" else 0
exang_val = 1 if exang == "Yes" else 0

user_data = pd.DataFrame({
    "age": [age],
    "sex": [sex_val],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs_val],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang_val],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})

user_scaled = scaler.transform(user_data)

# ---------------------- Prediction Box ---------------------
st.markdown("<div class='box'>", unsafe_allow_html=True)

st.subheader("🔍 Prediction")

if st.button("Predict"):
    prediction = model.predict(user_scaled)[0]
    if prediction == 1:
        st.error("🛑 The model predicts **HIGH RISK of Heart Disease**.")
    else:
        st.success("✅ The model predicts **NO Heart Disease Risk**.")

st.markdown("</div>", unsafe_allow_html=True)