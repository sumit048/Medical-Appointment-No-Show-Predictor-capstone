import pandas as pd
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -------------------- Load & Preprocess --------------------
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    df['DaysBetween'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df = df[df['DaysBetween'] >= 0]

    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
    df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

    # Keep PatientID if available
    if 'PatientId' in df.columns:
        df.rename(columns={'PatientId': 'PatientID'}, inplace=True)

    # Drop irrelevant columns for training
    df.drop(['AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood'], axis=1, inplace=True)
    return df

# Load and prepare data
data = load_and_preprocess("data/medical_no_show_data.csv")

# Split features and target
X = data.drop("No-show", axis=1)
y = data["No-show"]

# ✅ Remove PatientID from features if present
if "PatientID" in X.columns:
    X.drop(columns=["PatientID"], inplace=True)

# -------------------- Resample using SMOTE --------------------
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train/test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# -------------------- Train XGBoost Classifier --------------------
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# -------------------- Save Model and Feature Order --------------------
os.makedirs("model", exist_ok=True)

with open("model/no_show_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/feature_order.json", "w") as f:
    json.dump(list(X.columns), f)

print("✅ Model retrained with SMOTE + XGBoost and saved successfully!")
