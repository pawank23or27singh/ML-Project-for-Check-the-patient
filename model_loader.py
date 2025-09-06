import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

class DiseasePredictor:
    def __init__(self, models_dir: str = 'trained_models'):
        """
        Initialize the DiseasePredictor with the directory containing trained models.
        
        Args:
            models_dir: Directory containing the trained model files
        """
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self) -> None:
        """
        Load all available models and their corresponding scalers.
        Only loads each model once, even if called multiple times.
        """
        if not os.path.exists(self.models_dir):
            print(f"Warning: Models directory '{self.models_dir}' not found.")
            return
            
        # List all model files
        model_files = [f for f in os.listdir(self.models_dir) 
                      if f.endswith('.joblib') and 'scaler' not in f]
        
        for model_file in model_files:
            # Extract disease name from filename (e.g., 'diabetes_randomforest.joblib' -> 'diabetes')
            disease_name = model_file.split('_')[0].lower()
            
            # Skip if already loaded
            if disease_name in self.models:
                continue
                
            model_path = os.path.join(self.models_dir, model_file)
            scaler_path = os.path.join(self.models_dir, f"{disease_name}_scaler.joblib")
            
            try:
                # Load model
                model = joblib.load(model_path)
                
                # Handle case where loaded object is a list or tuple
                if isinstance(model, (list, tuple)) and len(model) > 0:
                    print(f"Warning: Found list/tuple for {disease_name} model. Using first element.")
                    model = model[0]
                
                # Verify the loaded object has predict method
                if not hasattr(model, 'predict'):
                    raise ValueError(f"Loaded object for {disease_name} does not have a predict method")
                
                self.models[disease_name] = model
                
                # Load corresponding scaler if exists
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    # Handle case where scaler is a list
                    if isinstance(scaler, (list, tuple)) and len(scaler) > 0:
                        print(f"Warning: Found list/tuple for {disease_name} scaler. Using first element.")
                        scaler = scaler[0]
                    self.scalers[disease_name] = scaler
                else:
                    # If no scaler found but model requires one, log a warning
                    if hasattr(model, 'feature_names_in_'):
                        print(f"Warning: Scaler not found for {disease_name} model. Some features may not be properly scaled.")
                
                print(f"Successfully loaded model: {disease_name} (Type: {type(model).__name__})")
                
            except Exception as e:
                print(f"Error loading {model_file}: {str(e)}")
                print(f"Model path: {model_path}")
                print(f"Model type: {type(model) if 'model' in locals() else 'N/A'}")
                # Remove the model if loading failed
                if disease_name in self.models:
                    del self.models[disease_name]
    
    def is_model_ready(self, disease: str) -> bool:
        """
        Check if a model is ready for prediction.
        
        Args:
            disease: Name of the disease model to check
            
        Returns:
            bool: True if the model is ready, False otherwise
        """
        disease = disease.lower()
        return disease in self.models
    
    def predict(self, disease: str, input_data: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Make a prediction for a specific disease.
        
        Args:
            disease: Name of the disease (e.g., 'diabetes', 'heart')
            input_data: Dictionary containing input features
            
        Returns:
            Tuple of (probability, confidence_scores)
            
        Raises:
            ValueError: If model is not found or not ready for prediction
        """
        disease = disease.lower()
        
        if not self.is_model_ready(disease):
            raise ValueError(f"Model for {disease} is not available or not properly loaded.")
        
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Get the model
            model = self.models[disease]
            
            # Check if model has feature names and input matches
            if hasattr(model, 'feature_names_in_'):
                missing_features = set(model.feature_names_in_) - set(input_data.keys())
                if missing_features:
                    raise ValueError(f"Missing required features: {', '.join(missing_features)}")
                
                # Reorder columns to match training data
                input_df = input_df[list(model.feature_names_in_)]
            
            # Apply preprocessing if scaler exists
            if disease in self.scalers:
                try:
                    input_scaled = self.scalers[disease].transform(input_df)
                except Exception as e:
                    print(f"Warning: Error scaling input for {disease}: {str(e)}")
                    input_scaled = input_df.values
            else:
                input_scaled = input_df.values
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled)[0]
                predicted_class = model.classes_[np.argmax(proba)]
                confidence = max(proba)
                
                # Create confidence scores for each class
                confidence_scores = {
                    str(cls): float(score) 
                    for cls, score in zip(model.classes_, proba)
                }
            else:
                # For models without predict_proba
                prediction = model.predict(input_scaled)[0]
                confidence = 1.0  # Default confidence
                confidence_scores = {str(prediction): confidence}
                
            return float(confidence), confidence_scores
            
        except Exception as e:
            print(f"Error during prediction for {disease}: {str(e)}")
            raise
    
    def get_available_models(self) -> list:
        """
        Return a list of available disease models that are ready for prediction.
        
        Returns:
            list: List of available model names
        """
        return list(self.models.keys())
    
    def get_model_features(self, disease: str) -> list:
        """
        Get the list of features required by a specific model.
        
        Args:
            disease: Name of the disease model
            
        Returns:
            list: List of feature names, or empty list if not available
        """
        disease = disease.lower()
        if disease in self.models and hasattr(self.models[disease], 'feature_names_in_'):
            return list(self.models[disease].feature_names_in_)
        return []
        
    def get_model_info(self, disease: str) -> dict:
        """
        Get detailed information about a specific model.
        
        Args:
            disease: Name of the disease model
            
        Returns:
            dict: Dictionary containing model information
        """
        disease = disease.lower()
        if disease not in self.models:
            return {"status": "not_found"}
            
        model = self.models[disease]
        info = {
            "status": "ready",
            "has_scaler": disease in self.scalers,
            "model_type": type(model).__name__,
            "features": self.get_model_features(disease),
            "n_features": len(self.get_model_features(disease)) if hasattr(model, 'feature_names_in_') else 0,
            "classes": model.classes_.tolist() if hasattr(model, 'classes_') else []
        }
        
        return info

# Create a global instance
try:
    predictor = DiseasePredictor()
except Exception as e:
    print(f"Error initializing DiseasePredictor: {str(e)}")
    predictor = None
