# 🩺 Medical Appointment No-Show Predictor

# Author

# Name:SUMIT MARADI

# Email:sumitmaradi85@gmail.com

This is a Streamlit-based machine learning app that predicts whether a patient will show up for their medical appointment.
This project predicts whether a patient will attend their medical appointment based on their demographic and health-related features. It uses machine learning (XGBoost) and provides a web interface built with Streamlit. If a patient is predicted to miss their appointment, an SMS alert is sent using Twilio.

## 🔹 Features

- Single patient prediction form with animated results
- Sending sms to no show patients
- Batch CSV upload support for bulk predictions
- Stylish UI with logo and dark mode support
- Downloadable batch result CSV in one click

## 🛠️ Setup Instructions

1. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the app:

```
streamlit run streamlit_app.py
```

## 📁 Files Included

- `streamlit_app.py`: Main Streamlit app
- `model/no_show_model.pkl`: Trained model file
- `model/feature_order.json`: Feature order file
- `doctor.jpg`: Background image (optional)
- `doctor_logo.jpg`: Logo image
- `batch_input_sample.csv`: Example input for batch prediction
- `batch_predictions_sample.csv`: Example output with predictions
- 'preview1.png': dashboard first page
- 'preview2.png': dashboard result
- 'preview3.png': dashboard batch prediction result

## 📊 Dataset

- Source: [Kaggle - Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
- Features include:
  - Age, Gender, Scholarship, Hypertension, Diabetes, Alcoholism, Handicap, SMS Received, DaysBetween (engineered)
- Target: `No-show` (1 = Did not show, 0 = Showed up)

## 📈 EDA Highlights

- Checked missing/null values
- Engineered `DaysBetween` = AppointmentDay - ScheduledDay
- Identified class imbalance
- Found that SMS reminders reduce no-shows

## 🧠 Model Training

- Model: `XGBoostClassifier`
- Used `SMOTE` to balance classes
- Saved trained model as `.pkl` and feature order as `.json`

## 💻 Streamlit Web App

- Responsive form to input patient data
- Predicts and displays result with animation
- ✅ "Likely to Show Up" or ❌ "Likely No Show up"
- Upload CSV file for batch predictions and download results

## 📲 Twilio SMS Integration

- Sends SMS to the patient **only if** they are predicted to miss the appointment
- Credentials are stored securely using `st.secrets`
- Trial account supports verified numbers

## 🚀 Deployment

- Hosted on Streamlit Cloud
- GitHub-connected repo with all source code and model files

## 🛡️ Exception Handling

- Errors during model load, file upload, and SMS are caught and displayed using `try-except`

## 📄 License

This project is for educational use as part of a capstone project.
