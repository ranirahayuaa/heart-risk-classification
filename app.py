import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Semua path relatif terhadap lokasi script ini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Page config
st.set_page_config(
    page_title="Heart Risk Classification",
    page_icon="❤️",
    layout="wide"
)

# Load model and preprocessors
@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, 'heart_risk_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    le_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')
    te_path = os.path.join(MODELS_DIR, 'target_encoder.pkl')
    fn_path = os.path.join(MODELS_DIR, 'feature_names.pkl')

    if not os.path.exists(model_path):
        st.error("❌ Model files not found! Please run 'python train.py' first to train the model.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(le_path)
    target_encoder = joblib.load(te_path)
    feature_names = joblib.load(fn_path) if os.path.exists(fn_path) else None
    return model, scaler, label_encoders, target_encoder, feature_names

model, scaler, label_encoders, target_encoder, feature_names = load_model()

# Risk level mapping
risk_colors = {
    "Low Risk": "#2ecc71",
    "Moderate Risk": "#f1c40f",
    "High Risk": "#e67e22",
    "Critical Risk": "#e74c3c"
}

# Header
st.title("❤️ Heart Risk Classification")
st.markdown("### Predict the risk level of heart disease based on patient health data")
st.markdown("---")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Information")

    # Input fields
    age = st.number_input("Age", min_value=18, max_value=92, value=50, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoker_status = st.selectbox("Smoker Status", ["Yes", "No"])
    diabetes_history = st.selectbox("Diabetes History", ["Yes", "No"])

    col_a, col_b = st.columns(2)

    with col_a:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=160, value=75, step=1)
        oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=80.0, max_value=100.0, value=98.0, step=0.1)

    with col_b:
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=230, value=120, step=1)
        body_temperature = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=39.0, value=36.5, step=0.1)

    chest_pain_severity = st.slider("Chest Pain Severity (0-10)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col2:
    st.subheader("Normal Ranges")
    st.markdown("""
    | Parameter | Normal Range |
    |-----------|--------------|
    | Heart Rate | 60-100 bpm |
    | Systolic BP | 90-120 mmHg |
    | Oxygen Sat | 95-100% |
    | Body Temp | 36.1-37.2°C |
    """)

    st.markdown("---")
    st.subheader("Key Risk Factors")
    st.markdown("""
    - ⚠️ Age > 60
    - ⚠️ Smoking
    - ⚠️ Diabetes
    - ⚠️ High Blood Pressure
    - ⚠️ High Heart Rate
    - ⚠️ Low Oxygen Saturation
    """)

# Predict button
if st.button("🔍 Predict Heart Risk", type="primary"):
    # Prepare input data
    input_data = {
        'age': [age],
        'gender': [label_encoders['gender'].transform([gender])[0]],
        'smoker_status': [label_encoders['smoker_status'].transform([smoker_status])[0]],
        'diabetes_history': [label_encoders['diabetes_history'].transform([diabetes_history])[0]],
        'heart_rate': [heart_rate],
        'systolic_blood_pressure': [systolic_bp],
        'oxygen_saturation': [oxygen_saturation],
        'chest_pain_severity': [chest_pain_severity],
        'body_temperature': [body_temperature]
    }

    input_df = pd.DataFrame(input_data)

    # Pastikan urutan kolom sama persis dengan saat training (mencegah error scaler)
    if feature_names is not None:
        input_df = input_df[feature_names]

    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    risk_level = target_encoder.inverse_transform([prediction])[0]
    confidence = prediction_proba[prediction] * 100

    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Risk Level", risk_level)

    with result_col2:
        st.metric("Confidence", f"{confidence:.2f}%")

    with result_col3:
        risk_color = risk_colors.get(risk_level, "#95a5a6")
        st.markdown(f"""
        <div style="background-color: {risk_color}; padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h3 style="margin: 0;">{risk_level}</h3>
        </div>
        """, unsafe_allow_html=True)

    # Probability chart
    st.markdown("### Risk Probability Distribution")

    classes = target_encoder.classes_
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=prediction_proba * 100,
            marker_color=[risk_colors.get(cls, "#95a5a6") for cls in classes],
            text=[f"{p*100:.2f}%" for p in prediction_proba],
            textposition='auto'
        )
    ])

    fig.update_layout(
        yaxis_title="Probability (%)",
        xaxis_title="Risk Level",
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(fig, use_container_width=True)

    # Recommendations
    st.markdown("### 📋 Recommendations")

    if risk_level == "Low Risk":
        st.success("✅ Keep up the healthy lifestyle! Continue regular check-ups.")
    elif risk_level == "Moderate Risk":
        st.warning("⚠️ Consult a healthcare provider. Consider lifestyle changes.")
    elif risk_level == "High Risk":
        st.error("🔴 Schedule a medical appointment immediately.")
    else:  # Critical Risk
        st.error("🚨 Seek emergency medical care immediately!")

    # Additional details
    with st.expander("ℹ️ Detailed Analysis"):
        st.markdown("**Confidence Scores:**")
        for i, risk in enumerate(classes):
            # st.progress(text=...) hanya tersedia di Streamlit >= 1.26
            try:
                st.progress(float(prediction_proba[i]), text=f"{risk}: {prediction_proba[i]*100:.2f}%")
            except TypeError:
                st.markdown(f"**{risk}:** {prediction_proba[i]*100:.2f}%")
                st.progress(float(prediction_proba[i]))

        st.markdown("\n**Input Summary:**")
        summary_data = {
            'Feature': ['Age', 'Gender', 'Smoker', 'Diabetes', 'Heart Rate',
                       'Systolic BP', 'Oxygen Sat', 'Chest Pain', 'Body Temp'],
            'Value': [age, gender, smoker_status, diabetes_history, heart_rate,
                     systolic_bp, oxygen_saturation, chest_pain_severity, body_temperature]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>Disclaimer: This is a predictive tool and should not replace professional medical advice.</small>
</div>
""", unsafe_allow_html=True)
