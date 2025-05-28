from typing import Dict, Any, List, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AdherencePredictor:
    """Predicts clinical trial adherence using multimodal data."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.prediction_threshold = 0.7
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
    
    def build_model(self, input_shape: Tuple[int, ...]) -> None:
        """Build the neural network model."""
        try:
            # Input layer
            inputs = layers.Input(shape=input_shape)
            
            # Feature extraction layers
            x = layers.Dense(128, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            # Temporal pattern recognition
            x = layers.Dense(64, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
            # Risk factor analysis
            x = layers.Dense(32, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
            # Output layer
            outputs = layers.Dense(1, activation='sigmoid')(x)
            
            # Create model
            self.model = models.Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.AUC()]
            )
            
            logger.info("Model built successfully")
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features from multimodal data."""
        try:
            features = []
            
            # Extract wearable device features
            if "wearable" in data:
                features.extend(self._extract_wearable_features(data["wearable"]))
            
            # Extract EHR features
            if "ehr" in data:
                features.extend(self._extract_ehr_features(data["ehr"]))
            
            # Extract survey features
            if "survey" in data:
                features.extend(self._extract_survey_features(data["survey"]))
            
            # Extract temporal patterns
            features.extend(self._extract_temporal_patterns(data))
            
            # Extract behavioral patterns
            features.extend(self._extract_behavioral_patterns(data))
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def _extract_wearable_features(self, wearable_data: Dict[str, Any]) -> List[float]:
        """Extract features from wearable device data."""
        features = []
        
        # Activity level features
        if "activity" in wearable_data:
            activity = wearable_data["activity"]
            features.extend([
                np.mean(activity),
                np.std(activity),
                np.max(activity),
                np.min(activity)
            ])
        
        # Sleep quality features
        if "sleep" in wearable_data:
            sleep = wearable_data["sleep"]
            features.extend([
                sleep.get("duration", 0),
                sleep.get("quality", 0),
                sleep.get("efficiency", 0)
            ])
        
        return features
    
    def _extract_ehr_features(self, ehr_data: Dict[str, Any]) -> List[float]:
        """Extract features from EHR data."""
        features = []
        
        # Medication adherence
        if "medications" in ehr_data:
            meds = ehr_data["medications"]
            features.extend([
                len(meds),
                sum(1 for m in meds if m.get("adherence", 0) > 0.8)
            ])
        
        # Visit attendance
        if "visits" in ehr_data:
            visits = ehr_data["visits"]
            features.extend([
                len(visits),
                sum(1 for v in visits if v.get("attended", False))
            ])
        
        return features
    
    def _extract_survey_features(self, survey_data: Dict[str, Any]) -> List[float]:
        """Extract features from survey data."""
        features = []
        
        # Symptom severity
        if "symptoms" in survey_data:
            symptoms = survey_data["symptoms"]
            features.extend([
                np.mean(symptoms),
                np.std(symptoms),
                np.max(symptoms)
            ])
        
        # Quality of life
        if "quality_of_life" in survey_data:
            qol = survey_data["quality_of_life"]
            features.extend([
                qol.get("physical", 0),
                qol.get("mental", 0),
                qol.get("social", 0)
            ])
        
        return features
    
    def _extract_temporal_patterns(self, data: Dict[str, Any]) -> List[float]:
        """Extract temporal patterns from the data."""
        features = []
        
        # Time-based patterns
        timestamps = []
        for source in ["wearable", "ehr", "survey"]:
            if source in data and "timestamp" in data[source]:
                timestamps.append(pd.to_datetime(data[source]["timestamp"]))
        
        if timestamps:
            # Calculate time differences
            time_diffs = np.diff(sorted(timestamps))
            features.extend([
                np.mean(time_diffs),
                np.std(time_diffs),
                np.max(time_diffs)
            ])
        
        return features
    
    def _extract_behavioral_patterns(self, data: Dict[str, Any]) -> List[float]:
        """Extract behavioral patterns from the data."""
        features = []
        
        # Activity patterns
        if "wearable" in data and "activity" in data["wearable"]:
            activity = data["wearable"]["activity"]
            # Calculate activity consistency
            features.append(np.std(activity))
        
        # Medication patterns
        if "ehr" in data and "medications" in data["ehr"]:
            meds = data["ehr"]["medications"]
            # Calculate medication consistency
            adherence_rates = [m.get("adherence", 0) for m in meds]
            features.append(np.mean(adherence_rates))
        
        return features
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> None:
        """Train the model on the provided data."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Build model if not already built
            if self.model is None:
                self.build_model(input_shape=(X.shape[1],))
            
            # Train model
            history = self.model.fit(
                X_scaled,
                y,
                validation_split=validation_split,
                epochs=50,
                batch_size=32,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Calculate feature importance
            self._calculate_feature_importance(X_scaled, y)
            
            logger.info("Model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate feature importance using permutation importance."""
        try:
            baseline_score = self.model.evaluate(X, y, verbose=0)[1]
            importance = []
            
            for i in range(X.shape[1]):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_score = self.model.evaluate(X_permuted, y, verbose=0)[1]
                importance.append(baseline_score - permuted_score)
            
            # Normalize importance scores
            importance = np.array(importance)
            importance = (importance - importance.min()) / (importance.max() - importance.min())
            
            self.feature_importance = dict(zip(range(X.shape[1]), importance))
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions on new data."""
        try:
            # Extract features
            features = self.extract_features(data)
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0][0]
            
            # Calculate confidence
            confidence = self._calculate_confidence(features_scaled)
            
            return {
                "risk_score": float(prediction),
                "confidence": float(confidence),
                "feature_importance": self.feature_importance,
                "threshold": self.prediction_threshold,
                "is_high_risk": prediction > self.prediction_threshold
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in the prediction."""
        try:
            # Use model's prediction probabilities
            probs = self.model.predict(features, verbose=0)[0]
            
            # Calculate confidence based on prediction probability
            confidence = 1 - abs(0.5 - probs[0]) * 2
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
    
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