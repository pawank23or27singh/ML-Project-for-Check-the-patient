from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

def create_asthma_model():
    # Create a simple random forest model for asthma prediction
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # We'll use these features for prediction:
    # Age, BMI, Smoking, Pollution Exposure, Pollen Exposure, 
    # Family History, Allergies, Wheezing, Shortness of Breath,
    # Chest Tightness, Coughing, Nighttime Symptoms, 
    # Exercise-Induced Symptoms, Lung Function FEV1, Lung Function FVC
    
    # Create dummy data for model training
    # This is just an example - in real application you would use actual patient data
    X = np.random.rand(100, 15)  # 100 samples with 15 features
    y = np.random.randint(0, 2, 100)  # Random labels (0 or 1)
    
    # Train the model
    model.fit(X, y)
    
    return model

# Create and save the model
model = create_asthma_model()

# Save the model
import joblib
joblib.dump(model, 'asthma_model.sav')
