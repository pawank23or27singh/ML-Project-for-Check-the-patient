import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

def load_asthma_data(filepath='asthma_disease_data.csv'):
    """Load and preprocess the asthma dataset."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
        
        # Basic preprocessing - this will need to be adjusted based on your actual data
        # For now, we'll create synthetic data if the file doesn't exist or can't be processed
        if df.empty or 'Asthma' not in df.columns:
            print("Warning: Invalid or empty dataset. Creating synthetic data...")
            return create_synthetic_data()
            
        # Separate features and target
        X = df.drop('Asthma', axis=1)
        y = df['Asthma']
        
        return X, y, list(X.columns)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data...")
        return create_synthetic_data()

def create_synthetic_data(n_samples=1000):
    """Create synthetic asthma data for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic features
    n_samples = 1000
    data = {
        'age': np.random.normal(45, 15, n_samples).clip(18, 90),
        'bmi': np.random.normal(26, 5, n_samples).clip(16, 50),
        'smoking': np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.25, 0.15]),  # 0=Never, 1=Former, 2=Current
        'pollution': np.random.randint(1, 11, n_samples),
        'pollen': np.random.randint(1, 11, n_samples),
        'family_history': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'allergies': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.2, 0.2, 0.2]),  # 0=None, 1=Seasonal, 2=Perennial, 3=Both
        'wheezing': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.5, 0.2, 0.2, 0.1]),  # 0=Never, 1=Rarely, 2=Sometimes, 3=Often
        'shortness_of_breath': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.5, 0.2, 0.2, 0.1]),  # 0=Never, 1=With Exercise, 2=At Rest, 3=Frequent
        'chest_tightness': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.5, 0.2, 0.2, 0.1]),  # 0=Never, 1=Occasional, 2=Frequent, 3=Constant
        'coughing': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.3, 0.2, 0.1]),  # 0=Never, 1=Occasional, 2=Frequent, 3=Constant
        'nighttime_symptoms': np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05]),  # 0=Never, 1=1-2x/month, 2=1-2x/week, 3=3-4x/week, 4=Daily
        'exercise_induced': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.5, 0.2, 0.2, 0.1]),  # 0=Never, 1=Rarely, 2=Sometimes, 3=Always
        'fev1': np.random.normal(85, 20, n_samples).clip(20, 150),  # % predicted
        'fvc': np.random.normal(90, 15, n_samples).clip(20, 150),   # % predicted
    }
    
    # Create target variable based on features (simplified for demo)
    X = pd.DataFrame(data)
    
    # Higher risk with: smoking, high pollution/pollen, family history, allergies, symptoms
    risk_factors = (
        (X['smoking'] == 2) * 0.3 +  # Current smoker
        (X['pollution'] > 7) * 0.2 +  # High pollution
        (X['pollen'] > 7) * 0.2 +     # High pollen
        (X['family_history'] == 1) * 0.3 +  # Family history
        (X['allergies'] > 0) * 0.2 +  # Any allergies
        (X[['wheezing', 'shortness_of_breath', 'chest_tightness', 'coughing']].sum(axis=1) > 4) * 0.5  # Multiple symptoms
    )
    
    # Add some noise
    risk_factors += np.random.normal(0, 0.1, n_samples)
    
    # Create binary target (1 if at risk of asthma)
    y = (risk_factors > 0.5).astype(int)
    
    # Balance the classes if needed
    if y.mean() < 0.3 or y.mean() > 0.7:
        # Adjust threshold to get more balanced classes
        threshold = np.percentile(risk_factors, 70 if y.mean() < 0.3 else 30)
        y = (risk_factors > threshold).astype(int)
    
    print(f"Generated synthetic dataset with {y.mean()*100:.1f}% positive cases")
    return X, y, list(X.columns)

def train_models(X, y, feature_names):
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
        'feature_names': feature_names,
        'scaler': scaler
    }

def save_model(model_info, output_dir='trained_models'):
    """Save the trained model and related files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the best model
    model_name = f"asthma_{model_info['best_model_name']}"
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model_info['best_model'], model_path)
    print(f"Saved model to {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(output_dir, f"asthma_scaler.joblib")
    joblib.dump(model_info['scaler'], scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(output_dir, f"asthma_features.joblib")
    joblib.dump(model_info['feature_names'], features_path)
    print(f"Saved feature names to {features_path}")
    
    # Save model info
    model_info_path = os.path.join(output_dir, f"asthma_model_info.txt")
    with open(model_info_path, 'w') as f:
        f.write(f"Asthma Prediction Model\n")
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
    print("=== Asthma Prediction Model Training ===")
    
    # Load data
    print("\nLoading data...")
    X, y, feature_names = load_asthma_data()
    
    # Train models
    print("\nTraining models...")
    model_info = train_models(X, y, feature_names)
    
    # Save the best model
    print("\nSaving model...")
    model_path = save_model(model_info)
    
    print("\n=== Training Complete ===")
    print(f"Best model: {model_info['best_model_name']}")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
