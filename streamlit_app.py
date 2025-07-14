import streamlit as st
import pandas as pd
import pickle
import json
import base64
from datetime import datetime
from twilio.rest import Client

def send_sms(to_number, message_body):
    account_sid = st.secrets["TWILIO_SID"]
    auth_token = st.secrets["TWILIO_TOKEN"]
    twilio_number = st.secrets["TWILIO_NUMBER"]
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=message_body,
        from_=twilio_number,
        to=to_number
    )
    return message.sid

# ------------------ PAGE CONFIGURATION ------------------
st.set_page_config(page_title="Medical Appointment No-Show Predictor", layout="wide")

def add_corner_logo(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .corner-logo {{
            position: fixed;
            top: 9px;
            left: 8px;
            width: 60px;
            z-index: 100;
        }}
        </style>
        <img class="corner-logo" src="data:image/png;base64,{encoded}">
        """,
        unsafe_allow_html=True
    )

add_corner_logo("doctor_logo.jpg")

# Load trained model
with open("model/no_show_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🩺 Medical Appointment No-Show Predictor")
st.header("Enter Patient Details")

# 🆕 Add Patient ID
patient_id = st.text_input("🆔 Patient ID", placeholder="Enter unique Patient ID")

# Inputs
age = st.number_input("Age", min_value=0, max_value=115, value=30)
days_between = st.slider("Days Between Scheduling and Appointment", 0, 50, 5)

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.radio("Gender", ["Female", "Male"])
with col2:
    scholarship = st.checkbox("Scholarship")
with col3:
    hypertension = st.checkbox("Hypertension")

col4, col5, col6 = st.columns(3)
with col4:
    diabetes = st.checkbox("Diabetes")
with col5:
    alcoholism = st.checkbox("Alcoholism")
with col6:
    handcap = st.selectbox("Handicap Level", [0, 1, 2, 3, 4])

sms_received = st.checkbox("SMS Received")
phone_number = st.text_input("📞 Patient Mobile Number (with country code)", placeholder="+91XXXXXXXXXX")

# Prepare Input
input_df = pd.DataFrame([{
    'Age': age,
    'Gender': 0 if gender == "Female" else 1,
    'Scholarship': int(scholarship),
    'Hipertension': int(hypertension),
    'Diabetes': int(diabetes),
    'Alcoholism': int(alcoholism),
    'Handcap': int(handcap),
    'SMS_received': int(sms_received),
    'DaysBetween': days_between
}])

# 🆕 Add Patient ID (for reference only)
input_df['PatientID'] = patient_id

# Predict Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Predict", use_container_width=True)

if predict_button:
    with open("model/feature_order.json", "r") as f:
        feature_order = json.load(f)

    input_for_model = input_df[feature_order]
    prediction = model.predict(input_for_model)[0]

    if prediction == 1:
        st.markdown("""
            <div style="background-color:#dc2626; padding:20px; border-radius:12px; text-align:center; animation: fadeIn 0.7s ease-out;">
                <img src="https://cdn-icons-png.flaticon.com/512/463/463612.png" width="80"/>
                <h3 style="color:white; margin-top:10px;">❌ Patient likely to NOT SHOW UP for the appointment</h3>
            </div>
        """, unsafe_allow_html=True)

        if phone_number.strip():
            try:
                send_sms(
                    to_number=phone_number.strip(),
                    message_body="🚨 Reminder: You might miss your medical appointment. Please confirm or reschedule."
                )
                timestamp = datetime.now().strftime("%I:%M %p")
                st.success(f"✅ SMS sent to {phone_number} at {timestamp}")
            except Exception as e:
                st.error(f"⚠️ Failed to send SMS: {e}")
        else:
            st.warning("⚠️ Please enter a valid phone number to send SMS.")
    else:
        st.markdown("""
            <div style="background-color:#22c55e; padding:20px; border-radius:12px; text-align:center; animation: fadeIn 0.7s ease-out;">
                <img src="https://cdn-icons-png.flaticon.com/512/845/845646.png" width="80"/>
                <h3 style="color:white; margin-top:10px;">✅ Patient likely to SHOW UP for the appointment</h3>
            </div>
        """, unsafe_allow_html=True)

# 📁 Batch Prediction Section
st.header("📁 Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    try:
        with open("model/feature_order.json", "r") as f:
            feature_order = json.load(f)

        X = df_uploaded[feature_order]
        preds = model.predict(X)

        df_uploaded["Prediction"] = preds
        df_uploaded["Prediction Result"] = df_uploaded["Prediction"].map({
            0: "✅ Will Show Up",
            1: "❌ Will Not Show Up"
        })

        st.dataframe(df_uploaded)

        csv = df_uploaded.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="batch_predictions.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
