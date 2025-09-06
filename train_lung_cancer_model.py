import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib
import os

def load_and_preprocess_data(filepath='lung_cancer.csv'):
    """Load and preprocess the lung cancer dataset."""
    try:
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Display basic info
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        
        return df, df.drop('LUNG_CANCER', axis=1), df['LUNG_CANCER']
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data...")
        return create_synthetic_data()

def create_synthetic_data(n_samples=1000):
    """Create synthetic lung cancer data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic features
    data = {
        'GENDER': np.random.choice(['M', 'F'], size=n_samples, p=[0.6, 0.4]),
        'AGE': np.random.normal(65, 10, n_samples).clip(40, 90).astype(int),
        'SMOKING': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        'YELLOW_FINGERS': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'ANXIETY': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
        'PEER_PRESSURE': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'CHRONIC_DISEASE': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
        'FATIGUE': np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5]),
        'ALLERGY': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
        'WHEEZING': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'ALCOHOL_CONSUMING': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
        'COUGHING': np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5]),
        'SHORTNESS_OF_BREATH': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
        'SWALLOWING_DIFFICULTY': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
        'CHEST_PAIN': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    }
    
    # Create target variable based on features (simplified for demo)
    df = pd.DataFrame(data)
    
    # Higher risk with: smoking, age, yellow fingers, chest pain, etc.
    risk_factors = (
        (df['SMOKING'] == 1) * 0.4 +  # Smoking
        (df['AGE'] > 60) * 0.3 +       # Older age
        (df['YELLOW_FINGERS'] == 1) * 0.2 +
        (df['CHEST_PAIN'] == 1) * 0.2 +
        (df['ALCOHOL_CONSUMING'] == 1) * 0.1
    )
    
    # Add some noise
    risk_factors += np.random.normal(0, 0.1, n_samples)
    
    # Create binary target (1 if at risk of lung cancer)
    y = (risk_factors > 0.5).astype(int)
    
    # Convert to YES/NO for consistency with original data
    df['LUNG_CANCER'] = np.where(y == 1, 'YES', 'NO')
    
    print(f"Generated synthetic dataset with {y.mean()*100:.1f}% positive cases")
    return df, df.drop('LUNG_CANCER', axis=1), df['LUNG_CANCER']

def preprocess_data(X, y):
    """Preprocess the data for model training."""
    # Encode categorical variables
    X_encoded = X.copy()
    
    # Encode gender
    if 'GENDER' in X_encoded.columns:
        X_encoded['GENDER'] = X_encoded['GENDER'].map({'M': 0, 'F': 1})
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X_encoded, y_encoded, le

def train_models(X, y):
    """Train and evaluate models."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models to try
    models = {
        'randomforest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'gradientboosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"{name.capitalize()} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
        
        # Save results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        # Track best model
        if f1 > best_score:
            best_score = f1
            best_model = name
    
    return {
        'best_model_name': best_model,
        'best_model': results[best_model]['model'],
        'results': results,
        'feature_names': X.columns.tolist(),
        'scaler': scaler
    }

def save_model(model_info, label_encoder, output_dir='trained_models'):
    """Save the trained model and related files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the best model
    model_name = f"lung_cancer_{model_info['best_model_name']}"
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model_info['best_model'], model_path)
    print(f"\nSaved model to {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, "lung_cancer_scaler.joblib")
    joblib.dump(model_info['scaler'], scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Save label encoder
    encoder_path = os.path.join(output_dir, "lung_cancer_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    print(f"Saved label encoder to {encoder_path}")
    
    # Save feature names
    features_path = os.path.join(output_dir, "lung_cancer_features.joblib")
    joblib.dump(model_info['feature_names'], features_path)
    print(f"Saved feature names to {features_path}")
    
    # Save model info
    model_info_path = os.path.join(output_dir, "lung_cancer_model_info.txt")
    with open(model_info_path, 'w') as f:
        f.write(f"Lung Cancer Prediction Model\n")
        f.write(f"Model Type: {model_info['best_model_name']}\n")
        f.write(f"Accuracy: {model_info['results'][model_info['best_model_name']]['accuracy']:.4f}\n")
        f.write(f"F1 Score: {model_info['results'][model_info['best_model_name']]['f1']:.4f}\n")
        f.write(f"ROC AUC: {model_info['results'][model_info['best_model_name']]['roc_auc']:.4f}\n\n")
        f.write("Feature Importances (if available):\n")
        
        if hasattr(model_info['best_model'], 'feature_importances_'):
            importances = model_info['best_model'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i, idx in enumerate(indices):
                if i < len(model_info['feature_names']):
                    f.write(f"  {model_info['feature_names'][idx]}: {importances[idx]:.4f}\n")
    
    print(f"Saved model info to {model_info_path}")
    return model_path

def main():
    print("=== Lung Cancer Prediction Model Training ===")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df, X, y = load_and_preprocess_data()
    X_encoded, y_encoded, label_encoder = preprocess_data(X, y)
    
    # Train models
    print("\nTraining models...")
    model_info = train_models(X_encoded, y_encoded)
    
    # Save the best model
    print("\nSaving model...")
    model_path = save_model(model_info, label_encoder)
    
    print("\n=== Training Complete ===")
    print(f"Best model: {model_info['best_model_name']}")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
