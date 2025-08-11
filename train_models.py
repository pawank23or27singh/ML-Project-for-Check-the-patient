#rough preview of the code to train diabetes and heart disease prediction models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Train Diabetes Prediction Model
def train_diabetes_model():
    # Load diabetes dataset
    df_diabetes = pd.read_csv('diabetes.csv')
    
    # Separate features and target
    X = df_diabetes.drop('Outcome', axis=1)
    y = df_diabetes['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
    diabetes_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = diabetes_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Diabetes Model Accuracy: {accuracy:.2f}")
    
    # Save the model
    with open('diabetes_model.sav', 'wb') as f:
        pickle.dump(diabetes_model, f)
    print("Diabetes model saved successfully")

# Train Heart Disease Prediction Model
def train_heart_model():
    # Load heart disease dataset
    df_heart = pd.read_csv('heart.csv')
    
    # Separate features and target
    X = df_heart.drop('target', axis=1)
    y = df_heart['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    heart_model = RandomForestClassifier(n_estimators=100, random_state=42)
    heart_model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = heart_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Heart Disease Model Accuracy: {accuracy:.2f}")
    
    # Save the model
    with open('heart_model.sav', 'wb') as f:
        pickle.dump(heart_model, f)
    print("Heart disease model saved successfully")

if __name__ == "__main__":
    print("Training Diabetes Model...")
    train_diabetes_model()
    
    print("\nTraining Heart Disease Model...")
    train_heart_model()
    
    print("\nAll models have been trained and saved!")
