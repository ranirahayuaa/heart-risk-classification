# ❤️ Heart Risk Classification

A machine learning web application for predicting heart disease risk levels based on patient health parameters.

---

## 🎯 Features

- **Interactive Web Interface** built with Streamlit
- **Machine Learning Model** using Random Forest Classifier
- **Risk Categories**: Low, Moderate, High, Critical
- **Real-time Predictions** with confidence scores
- **Visual Analytics**: Probability distribution charts & feature importance
- **Health Recommendations** based on risk level

---

## 📊 Input Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Age | 18-92 years | Patient age |
| Gender | Male/Female | Patient gender |
| Smoker Status | Yes/No | Current smoking status |
| Diabetes History | Yes/No | History of diabetes |
| Heart Rate | 40-160 bpm | Resting heart rate |
| Systolic BP | 70-230 mmHg | Systolic blood pressure |
| Oxygen Saturation | 80-100% | Blood oxygen level |
| Chest Pain Severity | 0-10 | Self-reported pain scale |
| Body Temperature | 35-39°C | Body temperature |

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ranirahayuaa/heart-risk-classification.git
cd heart-risk-classification

# Run the application
streamlit run app.py
