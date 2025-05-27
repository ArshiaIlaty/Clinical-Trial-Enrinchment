from typing import Dict, Any, List, Tuple
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AdherencePredictor:
    """Model for predicting clinical trial adherence."""
    
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.feature_names = [
            # Wearable features
            "heart_rate_resting",
            "heart_rate_variability",
            "sleep_duration",
            "sleep_quality",
            "activity_steps",
            "activity_calories",
            
            # EHR features
            "medication_count",
            "lab_result_count",
            "vital_sign_count",
            "clinical_note_count",
            
            # Survey features
            "adherence_score",
            "symptom_score",
            "quality_of_life_score",
            "side_effect_score"
        ]
    
    def _build_model(self) -> models.Model:
        """Build the neural network model."""
        model = models.Sequential([
            # Input layer
            layers.Dense(64, activation='relu', input_shape=(14,)),
            layers.Dropout(0.2),
            
            # Hidden layers
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def preprocess_features(
        self,
        wearable_data: Dict[str, Any],
        ehr_data: Dict[str, Any],
        survey_data: Dict[str, Any]
    ) -> np.ndarray:
        """Preprocess and combine features from different data sources."""
        try:
            features = []
            
            # Extract wearable features
            features.extend([
                wearable_data["heart_rate"]["resting"],
                wearable_data["heart_rate"]["variability"],
                wearable_data["sleep"]["duration"],
                wearable_data["sleep"]["quality"],
                wearable_data["activity"]["steps"],
                wearable_data["activity"]["calories"]
            ])
            
            # Extract EHR features
            features.extend([
                len(ehr_data["medications"]),
                len(ehr_data["lab_results"]),
                len(ehr_data["vitals"]),
                len(ehr_data["clinical_notes"])
            ])
            
            # Extract survey features
            survey_scores = survey_data.get("aggregate_scores", {})
            features.extend([
                survey_scores.get("adherence", 0.0),
                survey_scores.get("symptoms", 0.0),
                survey_scores.get("quality_of_life", 0.0),
                survey_scores.get("side_effects", 0.0)
            ])
            
            # Convert to numpy array and reshape
            features = np.array(features).reshape(1, -1)
            
            # Scale features
            features = self.scaler.fit_transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            raise
    
    def predict(
        self,
        wearable_data: Dict[str, Any],
        ehr_data: Dict[str, Any],
        survey_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make adherence prediction."""
        try:
            # Preprocess features
            features = self.preprocess_features(wearable_data, ehr_data, survey_data)
            
            # Make prediction
            prediction = self.model.predict(features)[0][0]
            
            # Calculate confidence based on feature importance
            confidence = self._calculate_confidence(features)
            
            return {
                "risk_score": float(prediction),
                "confidence": float(confidence),
                "timestamp": datetime.utcnow().isoformat(),
                "features": dict(zip(self.feature_names, features[0]))
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score based on feature importance and data quality."""
        try:
            # Get feature importance from the model
            feature_importance = self._get_feature_importance()
            
            # Calculate weighted average of feature values
            weighted_sum = np.sum(features * feature_importance)
            
            # Normalize to [0, 1] range
            confidence = (weighted_sum + 1) / 2
            
            return float(confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        # This is a placeholder implementation
        # In a real implementation, this would be calculated using SHAP values
        # or other feature importance methods
        return np.ones(len(self.feature_names)) / len(self.feature_names)
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """Train the model."""
        try:
            # Scale features
            X = self.scaler.fit_transform(X)
            
            # Train model
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            return {
                "loss": history.history["loss"],
                "accuracy": history.history["accuracy"],
                "val_loss": history.history["val_loss"],
                "val_accuracy": history.history["val_accuracy"]
            }
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def save_model(self, path: str) -> None:
        """Save the model to disk."""
        try:
            self.model.save(path)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str) -> None:
        """Load the model from disk."""
        try:
            self.model = models.load_model(path)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 