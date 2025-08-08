import streamlit as st
import requests
from PIL import Image

# This code is essential for the styling changes we discussed
try:
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ CSS file not found! Please create a '.streamlit/style.css' file.")
# === 🔗 API Endpoint ===
API_URL = "https://churn-prediction-api-9lrl.onrender.com/predict"

# === 🖼️ Add logo ===
# Replace 'logo.png' with your actual logo filename
try:
    logo = Image.open("assets/logo.jpeg")
    st.image(logo, width=200)
except FileNotFoundError:
    st.warning("⚠️ Logo not found! Please add 'logo.png' to the 'assets' folder.")

# === 🏷️ App Title ===
st.title('Customer Churn Prediction API')
st.markdown("### Predicting Customer Churn using FastAPI and MLOps Pipeline")

# === 📝 User Form ===
with st.form("churn_form"):
    st.header("Customer Information")

    customer_id = st.text_input("Customer ID", value="123456")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (in months)", 0, 72, 24)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", value=84.85)
    total_charges = st.number_input("Total Charges", value=1990.50)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    data = {
       # "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction Result: {result['prediction']}")
            st.metric("Churn Risk", f"{result['churn_risk_percent']}%")
            st.metric("Confidence", f"{result['confidence_percent']}%")
            st.write(f"**Will Churn:** {'Yes' if result['will_churn'] else 'No'}")
            st.write(f"**Risk Level:** {result['risk_level']}")
            st.write(f"**Confidence:** {result['confidence_percent']}%")
        else:
            st.error(f"Error from API: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")

st.markdown("---")
st.markdown("Built by Khoshaba Odeesho")