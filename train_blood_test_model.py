import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import joblib
import os

def load_blood_test_data(filepath='csb_blood_test_10000.csv'):
    """
    Load and preprocess the blood test data
    """
    try:
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Display basic info
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        
        return df, df.drop('condition', axis=1), df['condition']
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data...")
        return create_synthetic_data()

def create_synthetic_data(n_samples=1000):
    """Create synthetic blood test data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic features
    data = {
        'WBC': np.random.normal(7.5, 2.5, n_samples).clip(2, 15),  # White Blood Cells (10^9/L)
        'RBC': np.random.normal(4.8, 0.6, n_samples).clip(3.5, 6.0),  # Red Blood Cells (10^12/L)
        'HGB': np.random.normal(14.5, 1.5, n_samples).clip(12, 18),  # Hemoglobin (g/dL)
        'HCT': np.random.normal(43, 5, n_samples).clip(36, 50),  # Hematocrit (%)
        'MCV': np.random.normal(90, 8, n_samples).clip(80, 100),  # Mean Corpuscular Volume (fL)
        'MCH': np.random.normal(30, 3, n_samples).clip(27, 33),  # Mean Corpuscular Hemoglobin (pg)
        'MCHC': np.random.normal(34, 2, n_samples).clip(32, 36),  # Mean Corpuscular Hemoglobin Concentration (g/dL)
        'RDW': np.random.normal(13.5, 1.5, n_samples).clip(11.5, 15.5),  # Red Cell Distribution Width (%)
        'PLT': np.random.normal(250, 50, n_samples).clip(150, 450),  # Platelets (10^9/L)
        'MPV': np.random.normal(9.5, 1.5, n_samples).clip(7.5, 12.5),  # Mean Platelet Volume (fL)
        'PCT': np.random.normal(0.25, 0.05, n_samples).clip(0.15, 0.4),  # Plateletcrit (%)
        'PDW': np.random.normal(15, 2, n_samples).clip(10, 20),  # Platelet Distribution Width (%)
        'NEUT': np.random.normal(60, 10, n_samples).clip(40, 80),  # Neutrophils (%)
        'LYMPH': np.random.normal(30, 8, n_samples).clip(15, 45),  # Lymphocytes (%)
        'MONO': np.random.normal(7, 2, n_samples).clip(2, 12),  # Monocytes (%)
        'EO': np.random.normal(2.5, 1.5, n_samples).clip(0, 6),  # Eosinophils (%)
        'BASO': np.random.normal(0.5, 0.3, n_samples).clip(0, 2),  # Basophils (%)
    }
    
    # Create target variable based on features (simplified for demo)
    df = pd.DataFrame(data)
    
    # Higher risk with abnormal blood counts
    risk_factors = (
        ((df['WBC'] < 4) | (df['WBC'] > 11)) * 0.3 +  # Abnormal WBC
        ((df['HGB'] < 12) | (df['HGB'] > 16)) * 0.3 +  # Abnormal Hemoglobin
        ((df['PLT'] < 150) | (df['PLT'] > 400)) * 0.2 +  # Abnormal Platelets
        (df['LYMPH'] > 40) * 0.1 +  # High Lymphocytes
        (df['NEUT'] > 70) * 0.1  # High Neutrophils
    )
    
    # Add some noise
    risk_factors += np.random.normal(0, 0.1, n_samples)
    
    # Create binary target (1 if at risk)
    df['condition'] = (risk_factors > 0.5).astype(int)
    
    print(f"Generated synthetic dataset with {df['condition'].mean()*100:.1f}% positive cases")
    return df, df.drop('condition', axis=1), df['condition']

def preprocess_data(X, y):
    """Preprocess the data for model training."""
    # No need for encoding as all features are numerical
    return X, y, None

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
        print(classification_report(y_test, y_pred))
        
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
    model_name = f"blood_test_{model_info['best_model_name']}"
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model_info['best_model'], model_path)
    print(f"\nSaved model to {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, "blood_test_scaler.joblib")
    joblib.dump(model_info['scaler'], scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(output_dir, "blood_test_features.joblib")
    joblib.dump(model_info['feature_names'], features_path)
    print(f"Saved feature names to {features_path}")
    
    # Save model info
    model_info_path = os.path.join(output_dir, "blood_test_model_info.txt")
    with open(model_info_path, 'w') as f:
        f.write(f"Blood Test Analysis Model\n")
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
    print("=== Blood Test Analysis Model Training ===")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df, X, y = load_blood_test_data()
    X_processed, y_encoded, label_encoder = preprocess_data(X, y)
    
    # Train models
    print("\nTraining models...")
    model_info = train_models(X_processed, y_encoded)
    
    # Save the best model
    print("\nSaving model...")
    model_path = save_model(model_info, label_encoder)
    
    print("\n=== Training Complete ===")
    print(f"Best model: {model_info['best_model_name']}")
    print(f"Model saved to: {model_path}")
    
    # Print feature importances
    if hasattr(model_info['best_model'], 'feature_importances_'):
        print("\nTop 10 most important features:")
        importances = model_info['best_model'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i in range(min(10, len(model_info['feature_names']))):
            print(f"  {model_info['feature_names'][indices[i]]}: {importances[indices[i]]:.4f}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
