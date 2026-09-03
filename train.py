import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
import joblib
import warnings
import os
import matplotlib
matplotlib.use('Agg')  # backend headless agar tidak error tanpa display
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Semua path relatif terhadap lokasi script ini (aman dijalankan dari mana saja)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')


def find_dataset():
    """Cari dataset di beberapa lokasi umum."""
    candidates = [
        os.path.join(DATA_DIR, 'Heart_risk_dataset.csv'),
        os.path.join(BASE_DIR, 'Heart_risk_dataset.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def train():
    print("=" * 60)
    print("HEART RISK CLASSIFICATION - MODEL TRAINING")
    print("=" * 60)

    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Load data
    print("\n[1] Loading data...")
    dataset_path = find_dataset()
    if dataset_path is None:
        raise FileNotFoundError(
            "Heart_risk_dataset.csv tidak ditemukan. "
            "Letakkan di folder 'data/' atau sejajar dengan train.py"
        )
    print(f"    Using: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"    Data shape: {df.shape}")

    # Validasi kolom wajib
    required_cols = ['patient_id', 'age', 'gender', 'smoker_status', 'diabetes_history',
                     'heart_rate', 'systolic_blood_pressure', 'oxygen_saturation',
                     'chest_pain_severity', 'body_temperature', 'risk_level']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan di dataset: {missing}")

    # Drop patient_id
    df = df.drop('patient_id', axis=1)

    # Hapus baris dengan nilai kosong
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        print(f"    Dropped {before - len(df)} rows with missing values")

    # Encode categorical variables
    print(f"\n[2] Encoding categorical variables...")
    categorical_cols = ['gender', 'smoker_status', 'diabetes_history']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"    {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Encode target
    target_encoder = LabelEncoder()
    df['risk_level'] = target_encoder.fit_transform(df['risk_level'])
    print(f"\n    Target classes: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

    # Check class distribution
    print(f"\n[3] Class distribution:")
    class_counts = df['risk_level'].value_counts().sort_index()
    for i, count in class_counts.items():
        print(f"    {target_encoder.classes_[i]}: {count} ({count/len(df)*100:.1f}%)")

    # Split features and target
    X = df.drop('risk_level', axis=1)
    y = df['risk_level']

    # Split data
    print(f"\n[4] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train shape: {X_train.shape}")
    print(f"    Test shape: {X_test.shape}")

    # Scale numerical features
    print(f"\n[5] Scaling numerical features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print(f"\n[6] Training Random Forest model...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = rf.predict(X_test_scaled)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print("              Predicted")
    print("              " + " ".join([f"{cls:>10}" for cls in target_encoder.classes_]))
    for i, cls in enumerate(target_encoder.classes_):
        print(f"Actual {cls:>10}: " + " ".join([f"{val:>10}" for val in cm[i]]))

    # Save confusion matrix as image
    print(f"\n[7] Saving confusion matrix image...")
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_encoder.classes_)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Heart Risk Classification', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"    Confusion matrix saved to: images/confusion_matrix.png")
    plt.close()

    # Feature Importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"{row['feature']:>25}: {row['importance']:.4f} {bar}")

    # Save feature importance image
    print(f"\n[8] Saving feature importance image...")
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_data = feature_importance.head(10)
    bars = ax.barh(importance_data['feature'], importance_data['importance'], color='steelblue')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    print(f"    Feature importance saved to: images/feature_importance.png")
    plt.close()

    # Cross-validation
    print(f"\n[9] Performing cross-validation...")
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
    print(f"    Cross-validation mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Save model and preprocessors
    print(f"\n[10] Saving model and preprocessors...")
    joblib.dump(rf, os.path.join(MODELS_DIR, 'heart_risk_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    joblib.dump(target_encoder, os.path.join(MODELS_DIR, 'target_encoder.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, 'feature_names.pkl'))

    print(f"    Model saved to: models/heart_risk_model.pkl")
    print(f"    Scaler saved to: models/scaler.pkl")
    print(f"    Encoders saved to: models/label_encoders.pkl")
    print(f"    Target encoder saved to: models/target_encoder.pkl")
    print(f"    Feature names saved to: models/feature_names.pkl")

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        train()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
