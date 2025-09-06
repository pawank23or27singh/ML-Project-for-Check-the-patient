import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class DiseaseModelTrainer:
    def __init__(self):
        self.models_dir = "trained_models"
        self.results_dir = "model_results"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define dataset configurations
        self.dataset_configs = {
            'diabetes.csv': {
                'target': 'Outcome',
                'categorical': [],
                'numerical': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            },
            'heart.csv': {
                'target': 'target',
                'categorical': [],
                'numerical': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            },
            'parkinsons.csv': {
                'target': 'status',
                'categorical': [],
                'numerical': ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
                            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
                            'spread2', 'D2', 'PPE']
            },
            'lung_cancer.csv': {
                'target': 'LUNG_CANCER',
                'categorical': ['GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                              'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY',
                              'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
                              'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'],
                'numerical': ['AGE']
            },
            'stroke-data.csv': {
                'target': 'stroke',
                'categorical': ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'],
                'numerical': ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
            },
            'ChronicKidneyDisease_EHRs_from_AbuDhabi.csv': {
                'target': 'EventCKD35',
                'categorical': ['SEX', 'DM', 'HTN', 'IHD', 'CVA', 'Smoking', 'DLP', 'FH_ESRD',
                              'FH_CVD', 'FH_DM', 'FH_HTN', 'FH_ESL', 'FH_Cancer'],
                'numerical': ['Age', 'BMI', 'SBP', 'DBP', 'HR', 'FBS', 'HbA1c', 'UACR',
                            'eGFR', 'HGB', 'WBC', 'RBC', 'HCT', 'MCV', 'MCH', 'MCHC',
                            'RDW', 'PLT', 'LYM', 'MON', 'NEU', 'EOS', 'BAS', 'LYMp',
                            'MONp', 'NEUp', 'EOp', 'BAp', 'ALT', 'AST', 'ALP', 'TBIL',
                            'DBIL', 'IBIL', 'TP', 'ALB', 'GLOB', 'A_G', 'UREA', 'CREA',
                            'UA', 'CHOL', 'TG', 'HDL', 'LDL', 'VLDL', 'CA', 'PHOS',
                            'MG', 'NA', 'K', 'CL', 'CO2', 'PTH', 'VITD', 'URIC_ACID',
                            'URINE_PROTEIN', 'URINE_GLUCOSE', 'URINE_BLOOD', 'URINE_KETONES',
                            'URINE_LEUKOCYTES', 'URINE_NITRITE', 'URINE_BILIRUBIN',
                            'URINE_UBG', 'URINE_PH', 'URINE_SG', 'URINE_GLUCOSE_LEVEL']
            },
            'asthma_disease_data.csv': {
                'target': 'Asthma',
                'categorical': ['Gender', 'Allergies', 'Smoking', 'Exercise_Induced',
                              'Family_History', 'Cough', 'Wheezing', 'Shortness_of_Breath',
                              'Chest_Tightness'],
                'numerical': ['Age', 'FEV1', 'FVC', 'FEV1_FVC_Ratio', 'Peak_Exp_Flow',
                            'Pollen_Count', 'Dust_Level', 'Air_Quality_Index',
                            'Humidity', 'Temperature']
            },
            'alzheimers_synthetic.csv': {
                'target': 'Diagnosis',
                'categorical': ['M/F', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'],
                'numerical': ['Age', 'EDUC', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
            }
        }
        
    def load_data(self, filepath):
        """Load and preprocess the dataset"""
        print(f"\n{'='*50}")
        print(f"Processing: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None
    
    def preprocess_data(self, df, config):
        """Preprocess the data based on configuration"""
        # Drop rows with missing target values
        df = df.dropna(subset=[config['target']])
        
        # Convert target to binary if needed
        if df[config['target']].dtype == 'object':
            le = LabelEncoder()
            df[config['target']] = le.fit_transform(df[config['target']])
            
        # Separate features and target
        X = df.drop(columns=[config['target']])
        y = df[config['target']]
        
        # Handle categorical features
        categorical_features = [col for col in config['categorical'] if col in X.columns]
        if categorical_features:
            X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        
        # Handle missing values in numerical features
        numerical_features = [col for col in config['numerical'] if col in X.columns]
        for col in numerical_features:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        return X, y
    
    def train_and_evaluate(self, X, y, dataset_name):
        """Train and evaluate the model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to train
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
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
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = None
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'model': model
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc if roc_auc else 'N/A'}")
            
            # Update best model
            if accuracy > best_score:
                best_score = accuracy
                best_model = (name, model)
        
        # Save the best model
        if best_model:
            model_name, model = best_model
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{self.models_dir}/{dataset_name.replace('.csv', '')}_{model_name.lower()}.joblib"
            scaler_filename = f"{self.models_dir}/{dataset_name.replace('.csv', '')}_scaler.joblib"
            
            joblib.dump(model, model_filename)
            joblib.dump(scaler, scaler_filename)
            
            print(f"\nBest model saved: {model_filename}")
            print(f"Scaler saved: {scaler_filename}")
            
            # Save feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                feature_importance.to_csv(
                    f"{self.results_dir}/{dataset_name.replace('.csv', '')}_feature_importance.csv",
                    index=False
                )
                
                # Plot feature importance
                plt.figure(figsize=(10, 6))
                sns.barplot(x='importance', y='feature', 
                           data=feature_importance.head(15))
                plt.title(f'Top 15 Important Features - {dataset_name}')
                plt.tight_layout()
                plt.savefig(f"{self.results_dir}/{dataset_name.replace('.csv', '')}_feature_importance.png")
                plt.close()
        
        return results
    
    def run(self):
        """Run the training pipeline for all datasets"""
        all_results = {}
        
        for dataset in os.listdir('.'):
            if dataset in self.dataset_configs and dataset.endswith('.csv'):
                df = self.load_data(dataset)
                if df is not None:
                    try:
                        X, y = self.preprocess_data(df, self.dataset_configs[dataset])
                        results = self.train_and_evaluate(X, y, dataset)
                        all_results[dataset] = results
                    except Exception as e:
                        print(f"Error processing {dataset}: {str(e)}")
        
        # Save overall results
        if all_results:
            summary = []
            for dataset, models in all_results.items():
                for model_name, metrics in models.items():
                    summary.append({
                        'dataset': dataset,
                        'model': model_name,
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1': metrics['f1'],
                        'roc_auc': metrics['roc_auc']
                    })
            
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(f"{self.results_dir}/model_performance_summary.csv", index=False)
            
            print("\nTraining completed!")
            print("\nModel Performance Summary:")
            print(summary_df)
            
            return summary_df
        else:
            print("\nNo models were trained successfully.")
            return None

if __name__ == "__main__":
    trainer = DiseaseModelTrainer()

    trainer.run()
