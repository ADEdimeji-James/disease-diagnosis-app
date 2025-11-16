import os

# =========================
# Clean old model files
# =========================
for file in ["rf_tuned.pkl", "xgb_tuned.pkl", "lgb_tuned.pkl", 
             "label_encoder.pkl", "feature_list.json"]:
    if os.path.exists(file):
        os.remove(file)


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown
import os
import json
from sklearn.preprocessing import LabelEncoder

# =========================
# Download Models from Google Drive
# =========================
def download_file(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Google Drive Direct Links
files_to_download = {
    "rf_tuned.pkl": "https://drive.google.com/file/d/1xaNNx4WLivfNVi5dEX5XUYqEuKewiw3P/view?usp=sharing",
    "xgb_tuned.pkl": "https://drive.google.com/file/d/1aM5IBsE6XP6URZp_x5EG3RHOIZbLUgpM/view?usp=sharing",
    "lgb_tuned.pkl": "https://drive.google.com/file/d/115xw51iXHNV8QZJC7TmCZvxmm5P_jpsD/view?usp=sharing",
    "label_encoder.pkl": "https://drive.google.com/file/d/187bkRL9-lNGjC0wNuLW7734Bv1txXXSD/view?usp=sharing",
    "feature_list.json": "https://drive.google.com/file/d/1a94Qwnin0O1e5CJsIDwb0Zj_GnFFWYw1/view?usp=sharing"
}

for file_name, url in files_to_download.items():
    download_file(url, file_name)

# =========================
# Load Models + Encoders
# =========================
rf_model = joblib.load("rf_tuned.pkl")
xgb_model = joblib.load("xgb_tuned.pkl")
lgb_model = joblib.load("lgb_tuned.pkl")
label_encoder: LabelEncoder = joblib.load("label_encoder.pkl")

with open("feature_list.json", "r") as f:
    feature_list = json.load(f)

# Sort symptoms alphabetically for UI
feature_list_sorted = sorted(feature_list)

# =========================
# Streamlit App UI
# =========================
st.set_page_config(page_title="Disease Diagnosis App", layout="wide")
st.title("🩺 Disease Diagnosis App")
st.write("Select symptoms and get predicted disease using an ensemble of ML models.")

# Sidebar for symptoms
st.sidebar.header("Select Patient Symptoms")
selected_symptoms = []
for symptom in feature_list_sorted:
    if st.sidebar.checkbox(symptom):
        selected_symptoms.append(symptom)

# Create input vector
input_vector = np.zeros(len(feature_list))
for symptom in selected_symptoms:
    idx = feature_list.index(symptom)
    input_vector[idx] = 1

# =========================
# Prediction Logic
# =========================
if st.sidebar.button("Predict Disease"):
    input_vector = input_vector.reshape(1, -1)

    # Model probabilities
    rf_probs = rf_model.predict_proba(input_vector)
    xgb_probs = xgb_model.predict_proba(input_vector)
    lgb_probs = lgb_model.predict_proba(input_vector)

    # Soft voting (average probabilities)
    avg_probs = (rf_probs + xgb_probs + lgb_probs) / 3
    predicted_class_idx = np.argmax(avg_probs)
    predicted_disease = label_encoder.inverse_transform([predicted_class_idx])[0]

    # Top 3 predictions
    top3_idx = np.argsort(avg_probs[0])[::-1][:3]
    top3_diseases = label_encoder.inverse_transform(top3_idx)
    top3_probs = avg_probs[0][top3_idx]

    st.subheader("✅ Predicted Disease")
    st.success(f"{predicted_disease}")

    st.subheader("📊 Prediction Confidence (Top 3)")
    results_df = pd.DataFrame({
        "Disease": top3_diseases,
        "Probability": np.round(top3_probs * 100, 2)
    })
    st.table(results_df)

else:
    st.info("👉 Select symptoms from the sidebar and click **Predict Disease**")


