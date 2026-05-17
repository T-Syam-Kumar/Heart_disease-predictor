#  Heart Disease Prediction App

[**Live Demo → heartdisease‑predictor‑31.streamlit.app**](https://heartdisease-predictor-31.streamlit.app/)

##  What is this?

This is a simple and intuitive web app built with **Streamlit + Python + scikit‑learn** to predict the likelihood of heart disease, given some basic medical parameters (age, blood pressure, cholesterol, ECG results, etc.). It uses a trained logistic‑regression model on a publicly available “heart disease” dataset to output a clear prediction:

- ✅ **No Heart Disease Risk**  
- 🛑 **High Risk of Heart Disease**
 
Use this to quickly check risk based on patient data — no installation, just open the link and fill in the form!

##  Why it matters

Heart disease remains one of the leading health concerns worldwide. Early detection and preventive assessment can make a big difference. This app aims to:

- Offer a **quick risk‑assessment tool** using basic patient information  
- Enable users (doctors, patients, enthusiasts) to **explore heart‑disease likelihood without needing ML expertise**  
- Serve as a **learning/demo project** for how to deploy machine‑learning models using Streamlit  

---

##  Features

-  Clean and user‑friendly UI built with Streamlit  
-  Accepts standard medical inputs (age, blood pressure, cholesterol, ECG, etc.)  
-  Instant prediction using a pre‑trained logistic regression model  
-  Sidebar shows the model details and accuracy on test data  
-  Option to view the original dataset

---

##  How to Run Locally

```bash
git clone https://github.com/T-Syam-Kumar/Heart_disease-predictor.git
cd Heart_disease-predictor
python -m venv venv
# On Windows
venv\\Scripts\\activate
# On macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
