import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
import joblib
import pickle
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os
import json
import uuid
from motor.motor_asyncio import AsyncIOMotorCollection
from models import MLModelMetadata, EXOPLANET_TYPES

logger = logging.getLogger(__name__)

class MLEngine:
    """
    Comprehensive machine learning engine for exoplanet classification.
    Supports multiple algorithms and ensemble methods.
    """
    
    def __init__(self, db_collection: AsyncIOMotorCollection):
        self.db_collection = db_collection
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.target_classes = []
        self.model_storage_path = "/tmp/models"
        
        # Ensure model storage directory exists
        os.makedirs(self.model_storage_path, exist_ok=True)
        
        # Define algorithm configurations
        self.algorithm_configs = {
            'Random Forest': {
                'class': RandomForestClassifier,
                'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
            },
            'XGBoost': {
                'class': XGBClassifier,
                'params': {'n_estimators': 100, 'random_state': 42, 'eval_metric': 'mlogloss'}
            },
            'SVM': {
                'class': SVC,
                'params': {'kernel': 'rbf', 'random_state': 42, 'probability': True}
            },
            'Logistic Regression': {
                'class': LogisticRegression,
                'params': {'random_state': 42, 'max_iter': 1000}
            },
            'Extra Trees': {
                'class': ExtraTreesClassifier,
                'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
            },
            'Gradient Boosting': {
                'class': GradientBoostingClassifier,
                'params': {'n_estimators': 100, 'random_state': 42}
            }
        }
    
    async def prepare_training_data(self, 
                                 data: pd.DataFrame,
                                 feature_columns: List[str],
                                 target_column: str = 'exoplanet_type',
                                 handle_missing: str = 'median',
                                 apply_scaling: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Prepare data for ML training."""
        try:
            logger.info(f"Preparing training data with {len(data)} samples")
            
            # Filter for available feature columns
            available_features = [col for col in feature_columns if col in data.columns]
            if not available_features:
                raise ValueError("No valid feature columns found in data")
            
            # Handle missing values
            df_clean = data.copy()
            
            if handle_missing == 'drop':
                df_clean = df_clean.dropna(subset=available_features + [target_column])
            elif handle_missing == 'median':
                for col in available_features:
                    if df_clean[col].dtype in ['float64', 'int64']:
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
            elif handle_missing == 'mean':
                for col in available_features:
                    if df_clean[col].dtype in ['float64', 'int64']:
                        mean_val = df_clean[col].mean()
                        df_clean[col] = df_clean[col].fillna(mean_val)
            
            # Remove rows with missing target values
            df_clean = df_clean.dropna(subset=[target_column])
            
            if len(df_clean) == 0:
                raise ValueError("No valid samples remaining after cleaning")
            
            # Prepare features and target
            X = df_clean[available_features].values
            y = df_clean[target_column].values
            
            # Encode target labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Apply feature scaling if requested
            scaler = None
            if apply_scaling:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            # Store preprocessing objects
            self.feature_columns = available_features
            self.target_classes = label_encoder.classes_.tolist()
            
            preprocessing_info = {
                'feature_columns': available_features,
                'target_classes': self.target_classes,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'samples_count': len(df_clean),
                'features_count': len(available_features)
            }
            
            logger.info(f"Data preparation complete: {len(df_clean)} samples, {len(available_features)} features")
            return X, y_encoded, preprocessing_info
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    async def train_model(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         model_type: str,
                         preprocessing_info: Dict[str, Any],
                         test_size: float = 0.2,
                         random_state: int = 42,
                         perform_cv: bool = True) -> Dict[str, Any]:
        """Train a single ML model."""
        try:
            logger.info(f"Training {model_type} model")
            
            if model_type not in self.algorithm_configs:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Initialize model
            model_config = self.algorithm_configs[model_type]
            model = model_config['class'](**model_config['params'])
            
            # Handle Neural Network separately
            if model_type == 'Neural Network':
                model = await self._train_neural_network(X_train, y_train, X_test, y_test, preprocessing_info)
            else:
                # Train model
                model.fit(X_train, y_train)
            
            # Make predictions
            if model_type == 'Neural Network':
                y_pred = np.argmax(model.predict(X_test), axis=1)
                y_pred_proba = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Cross-validation
            cv_scores = None
            if perform_cv and model_type != 'Neural Network':
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(preprocessing_info['feature_columns'], model.feature_importances_))
                feature_importance = importance_dict
            elif hasattr(model, 'coef_') and model_type == 'Logistic Regression':
                # For logistic regression, use mean of absolute coefficients
                coef_mean = np.mean(np.abs(model.coef_), axis=0)
                importance_dict = dict(zip(preprocessing_info['feature_columns'], coef_mean))
                feature_importance = importance_dict
            
            # Save model and scaler
            model_id = str(uuid.uuid4())
            model_path = os.path.join(self.model_storage_path, f"model_{model_id}.joblib")
            scaler_path = os.path.join(self.model_storage_path, f"scaler_{model_id}.joblib")
            
            if model_type == 'Neural Network':
                model_path = os.path.join(self.model_storage_path, f"model_{model_id}.h5")
                model.save(model_path)
            else:
                joblib.dump(model, model_path)
            
            if preprocessing_info['scaler'] is not None:
                joblib.dump(preprocessing_info['scaler'], scaler_path)
            else:
                scaler_path = None
            
            # Store model in memory
            self.models[model_id] = {
                'model': model,
                'model_type': model_type,
                'scaler': preprocessing_info['scaler'],
                'label_encoder': preprocessing_info['label_encoder'],
                'feature_columns': preprocessing_info['feature_columns'],
                'target_classes': preprocessing_info['target_classes']
            }
            
            # Performance metrics
            performance_metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'cv_mean': float(cv_scores.mean()) if cv_scores is not None else 0.0,
                'cv_std': float(cv_scores.std()) if cv_scores is not None else 0.0,
                'test_samples': int(len(y_test)),
                'train_samples': int(len(y_train))
            }
            
            # Create metadata
            metadata = MLModelMetadata(
                id=model_id,
                name=f"{model_type} Exoplanet Classifier",
                model_type=model_type,
                training_data_count=len(X),
                features=preprocessing_info['feature_columns'],
                target_classes=preprocessing_info['target_classes'],
                performance_metrics=performance_metrics,
                model_file_path=model_path,
                scaler_file_path=scaler_path
            )
            
            # Save to database
            await self.db_collection.insert_one(metadata.dict())
            
            result = {
                'model_id': model_id,
                'model_type': model_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_scores': cv_scores.tolist() if cv_scores is not None else [],
                'confusion_matrix': conf_matrix.tolist(),
                'feature_importance': feature_importance,
                'performance_metrics': performance_metrics,
                'classification_report': classification_report(y_test, y_pred, target_names=preprocessing_info['target_classes'], output_dict=True)
            }
            
            logger.info(f"Successfully trained {model_type} model (ID: {model_id}) - Accuracy: {accuracy:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {str(e)}")
            raise
    
    async def _train_neural_network(self, X_train, y_train, X_test, y_test, preprocessing_info):
        """Train a neural network model using TensorFlow/Keras."""
        try:
            # Convert to categorical
            num_classes = len(preprocessing_info['target_classes'])
            y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
            y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
            
            # Build model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train_cat,
                validation_data=(X_test, y_test_cat),
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ]
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error training neural network: {str(e)}")
            raise
    
    async def predict_single(self, 
                           features: Dict[str, float], 
                           model_id: Optional[str] = None,
                           model_type: Optional[str] = None) -> Dict[str, Any]:
        """Make prediction for a single sample."""
        try:
            # Load model if needed
            if model_id and model_id not in self.models:
                await self._load_model(model_id)
            elif model_type and not model_id:
                # Find best model of specified type
                model_id = await self._find_best_model(model_type)
                if not model_id:
                    raise ValueError(f"No trained model found for type: {model_type}")
            elif not model_id and not model_type:
                # Use best overall model
                model_id = await self._find_best_model()
                if not model_id:
                    raise ValueError("No trained models available")
            
            model_info = self.models[model_id]
            model = model_info['model']
            scaler = model_info['scaler']
            label_encoder = model_info['label_encoder']
            feature_columns = model_info['feature_columns']
            
            # Prepare features
            feature_values = [features.get(col, 0.0) for col in feature_columns]
            X = np.array([feature_values])
            
            # Apply scaling if available
            if scaler is not None:
                X = scaler.transform(X)
            
            # Make prediction
            if model_info['model_type'] == 'Neural Network':
                prediction_proba = model.predict(X)[0]
                prediction = np.argmax(prediction_proba)
                confidence = float(np.max(prediction_proba))
                probabilities = dict(zip(model_info['target_classes'], prediction_proba.astype(float)))
            else:
                prediction = model.predict(X)[0]
                prediction_proba = model.predict_proba(X)[0]
                confidence = float(np.max(prediction_proba))
                probabilities = dict(zip(model_info['target_classes'], prediction_proba.astype(float)))
            
            # Convert prediction back to class name
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            
            # Feature importance for this prediction (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, model.feature_importances_.astype(float)))
            
            result = {
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities,
                'model_used': model_id,
                'model_type': model_info['model_type'],
                'feature_importance': feature_importance
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making single prediction: {str(e)}")
            raise
    
    async def predict_batch(self,
                          features_list: List[Dict[str, float]],
                          model_id: Optional[str] = None,
                          model_type: Optional[str] = None,
                          use_ensemble: bool = False) -> Dict[str, Any]:
        """Make predictions for multiple samples."""
        try:
            if use_ensemble:
                return await self._ensemble_predict(features_list)
            
            # Load model if needed
            if model_id and model_id not in self.models:
                await self._load_model(model_id)
            elif model_type and not model_id:
                model_id = await self._find_best_model(model_type)
            elif not model_id and not model_type:
                model_id = await self._find_best_model()
            
            if not model_id:
                raise ValueError("No suitable model found")
            
            model_info = self.models[model_id]
            model = model_info['model']
            scaler = model_info['scaler']
            label_encoder = model_info['label_encoder']
            feature_columns = model_info['feature_columns']
            
            # Prepare features
            X = []
            for features in features_list:
                feature_values = [features.get(col, 0.0) for col in feature_columns]
                X.append(feature_values)
            X = np.array(X)
            
            # Apply scaling if available
            if scaler is not None:
                X = scaler.transform(X)
            
            # Make predictions
            if model_info['model_type'] == 'Neural Network':
                predictions_proba = model.predict(X)
                predictions = np.argmax(predictions_proba, axis=1)
                confidences = np.max(predictions_proba, axis=1)
            else:
                predictions = model.predict(X)
                predictions_proba = model.predict_proba(X)
                confidences = np.max(predictions_proba, axis=1)
            
            # Convert predictions back to class names
            predicted_classes = label_encoder.inverse_transform(predictions).tolist()
            
            result = {
                'predictions': predicted_classes,
                'confidence': confidences.astype(float).tolist(),
                'model_used': model_id,
                'model_type': model_info['model_type'],
                'num_predictions': len(predicted_classes)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise
    
    async def _ensemble_predict(self, features_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """Make ensemble predictions using all available models."""
        try:
            if not self.models:
                raise ValueError("No models available for ensemble prediction")
            
            all_predictions = []
            model_weights = []
            
            for model_id, model_info in self.models.items():
                try:
                    # Get model performance weight (accuracy)
                    metadata = await self.db_collection.find_one({'id': model_id})
                    weight = metadata['performance_metrics']['accuracy'] if metadata else 1.0
                    
                    # Make predictions with this model
                    batch_result = await self.predict_batch(features_list, model_id=model_id)
                    all_predictions.append(batch_result['predictions'])
                    model_weights.append(weight)
                    
                except Exception as e:
                    logger.warning(f"Error using model {model_id} in ensemble: {str(e)}")
                    continue
            
            if not all_predictions:
                raise ValueError("No models successfully made predictions")
            
            # Weighted voting
            final_predictions = []
            confidences = []
            
            for i in range(len(features_list)):
                vote_counts = {}
                total_weight = 0
                
                for j, predictions in enumerate(all_predictions):
                    prediction = predictions[i]
                    weight = model_weights[j]
                    
                    if prediction not in vote_counts:
                        vote_counts[prediction] = 0
                    vote_counts[prediction] += weight
                    total_weight += weight
                
                # Get prediction with highest weighted vote
                best_prediction = max(vote_counts, key=vote_counts.get)
                confidence = vote_counts[best_prediction] / total_weight
                
                final_predictions.append(best_prediction)
                confidences.append(confidence)
            
            result = {
                'predictions': final_predictions,
                'confidence': confidences,
                'model_used': 'ensemble',
                'num_models': len(all_predictions),
                'models_used': list(self.models.keys())
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            raise
    
    async def _load_model(self, model_id: str):
        """Load a model from storage."""
        try:
            # Get model metadata
            metadata = await self.db_collection.find_one({'id': model_id})
            if not metadata:
                raise ValueError(f"Model metadata not found: {model_id}")
            
            # Load model
            model_path = metadata['model_file_path']
            if metadata['model_type'] == 'Neural Network':
                model = tf.keras.models.load_model(model_path)
            else:
                model = joblib.load(model_path)
            
            # Load scaler if available
            scaler = None
            if metadata['scaler_file_path']:
                scaler = joblib.load(metadata['scaler_file_path'])
            
            # Create label encoder
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(metadata['target_classes'])
            
            # Store in memory
            self.models[model_id] = {
                'model': model,
                'model_type': metadata['model_type'],
                'scaler': scaler,
                'label_encoder': label_encoder,
                'feature_columns': metadata['features'],
                'target_classes': metadata['target_classes']
            }
            
            logger.info(f"Successfully loaded model {model_id}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            raise
    
    async def _find_best_model(self, model_type: Optional[str] = None) -> Optional[str]:
        """Find the best model (highest accuracy) of specified type or overall."""
        try:
            query = {'is_active': True}
            if model_type:
                query['model_type'] = model_type
            
            # Find models sorted by accuracy descending
            cursor = self.db_collection.find(query).sort('performance_metrics.accuracy', -1).limit(1)
            models = await cursor.to_list(length=1)
            
            if models:
                return models[0]['id']
            return None
            
        except Exception as e:
            logger.error(f"Error finding best model: {str(e)}")
            return None
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models."""
        try:
            models = await self.db_collection.find({'is_active': True}).to_list(length=100)
            
            performance_data = {}
            for model in models:
                performance_data[model['id']] = {
                    'model_type': model['model_type'],
                    'name': model['name'],
                    'accuracy': model['performance_metrics']['accuracy'],
                    'precision': model['performance_metrics']['precision'],
                    'recall': model['performance_metrics']['recall'],
                    'f1_score': model['performance_metrics']['f1_score'],
                    'cv_mean': model['performance_metrics']['cv_mean'],
                    'cv_std': model['performance_metrics']['cv_std'],
                    'created_at': model['created_at'],
                    'training_data_count': model['training_data_count'],
                    'features': model['features']
                }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return {}
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model and its files."""
        try:
            # Get model metadata
            metadata = await self.db_collection.find_one({'id': model_id})
            if not metadata:
                return False
            
            # Delete model files
            if os.path.exists(metadata['model_file_path']):
                os.remove(metadata['model_file_path'])
            
            if metadata['scaler_file_path'] and os.path.exists(metadata['scaler_file_path']):
                os.remove(metadata['scaler_file_path'])
            
            # Remove from database
            await self.db_collection.delete_one({'id': model_id})
            
            # Remove from memory
            if model_id in self.models:
                del self.models[model_id]
            
            logger.info(f"Successfully deleted model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {str(e)}")
            return False