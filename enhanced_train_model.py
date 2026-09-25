import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
import argparse
import time
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           precision_score, recall_score, f1_score)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    def __init__(self, config=None):
        self.config = config or {
            'csv_path': 'processed_landmarks.csv',
            'model_path': 'sign_language_model.pkl',
            'results_dir': 'training_results',
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'enable_feature_selection': True,
            'enable_dimensionality_reduction': False,
            'enable_ensemble': True,
            'enable_hyperparameter_tuning': True,
            'cross_validation_folds': 5,
            'max_features_pca': 50,
            'scale_features': True
        }
        
        # Create results directory
        Path(self.config['results_dir']).mkdir(exist_ok=True)
        
        # Initialize components
        self.scaler = StandardScaler() if self.config['scale_features'] else None
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.pca = None
        self.best_model = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset with class balancing"""
        logger.info("Loading dataset...")
        
        # Load data
        df = pd.read_csv(self.config['csv_path'])
        logger.info(f"Loaded dataset with {len(df)} samples and {len(df.columns)-1} features")
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col != 'label']
        X = df[feature_cols].values
        y = df['label'].values
        
        # Handle missing values
        if np.any(np.isnan(X)):
            logger.warning("Found NaN values in features, filling with median")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Print class distribution
        class_counts = Counter(y)
        logger.info("Class distribution:")
        total_samples = len(y)
        for label, count in sorted(class_counts.items()):
            percentage = (count / total_samples) * 100
            logger.info(f"  {label}: {count} samples ({percentage:.1f}%)")
        
        # Handle class imbalance
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 3:
            logger.warning(f"Significant class imbalance detected (ratio: {imbalance_ratio:.2f})")
            logger.info("Applying class balancing strategy...")
            
            # Apply SMOTE for severe imbalance or class weights for moderate imbalance
            if imbalance_ratio > 10:
                try:
                    from imblearn.over_sampling import SMOTE
                    logger.info("Applying SMOTE oversampling...")
                    smote = SMOTE(random_state=self.config['random_state'], k_neighbors=min(5, min_count-1))
                    X, y_encoded = smote.fit_resample(X, y_encoded)
                    logger.info(f"Dataset size after SMOTE: {len(X)} samples")
                    
                    # Update class distribution
                    y = self.label_encoder.inverse_transform(y_encoded)
                    new_class_counts = Counter(y)
                    logger.info("New class distribution after SMOTE:")
                    for label, count in sorted(new_class_counts.items()):
                        percentage = (count / len(y)) * 100
                        logger.info(f"  {label}: {count} samples ({percentage:.1f}%)")
                        
                except ImportError:
                    logger.warning("imbalanced-learn not installed. Install with: pip install imbalanced-learn")
                    logger.info("Using class weights instead...")
                    self.use_class_weights = True
            else:
                logger.info("Using class weights to handle imbalance...")
                self.use_class_weights = True
        else:
            self.use_class_weights = False
        
        return X, y_encoded, feature_cols
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        logger.info("Splitting data...")
        
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        # Second split: train vs val
        val_size = self.config['validation_size'] / (1 - self.config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size,
            random_state=self.config['random_state'], stratify=y_train_val
        )
        
        logger.info(f"Dataset split - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_features(self, X_train, X_val, X_test):
        """Apply feature preprocessing"""
        logger.info("Preprocessing features...")
        
        # Scale features
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
            logger.info("Applied feature scaling")
        
        return X_train, X_val, X_test
    
    def feature_selection(self, X_train, y_train, X_val, X_test, n_features=100):
        """Perform feature selection"""
        if not self.config['enable_feature_selection']:
            return X_train, X_val, X_test
        
        logger.info("Performing feature selection...")
        
        # Use ANOVA F-test for feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(n_features, X_train.shape[1]))
        X_train = self.feature_selector.fit_transform(X_train, y_train)
        X_val = self.feature_selector.transform(X_val)
        X_test = self.feature_selector.transform(X_test)
        
        logger.info(f"Selected {X_train.shape[1]} features out of {len(self.feature_selector.get_support())}")
        
        return X_train, X_val, X_test
    
    def dimensionality_reduction(self, X_train, X_val, X_test):
        """Apply PCA for dimensionality reduction"""
        if not self.config['enable_dimensionality_reduction']:
            return X_train, X_val, X_test
        
        logger.info("Applying PCA...")
        
        n_components = min(self.config['max_features_pca'], X_train.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.config['random_state'])
        
        X_train = self.pca.fit_transform(X_train)
        X_val = self.pca.transform(X_val)
        X_test = self.pca.transform(X_test)
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA reduced to {n_components} components, explained variance: {explained_variance:.3f}")
        
        return X_train, X_val, X_test
    
    def define_models(self):
        """Define models to train with class balancing support"""
        # Determine if we need class weights
        class_weight = 'balanced' if hasattr(self, 'use_class_weights') and self.use_class_weights else None
        
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(
                    random_state=self.config['random_state'],
                    class_weight=class_weight
                ),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=self.config['random_state']),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                }
            },
            'SVM': {
                'model': SVC(
                    random_state=self.config['random_state'], 
                    probability=True,
                    class_weight=class_weight
                ),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            'MLP': {
                'model': MLPClassifier(
                    random_state=self.config['random_state'], 
                    max_iter=1000
                ),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        if class_weight:
            logger.info("Using balanced class weights to handle class imbalance")
        
        return models
    
    def train_single_model(self, name, model_config, X_train, y_train, X_val, y_val):
        """Train a single model with hyperparameter tuning"""
        logger.info(f"Training {name}...")
        
        model = model_config['model']
        param_grid = model_config['params']
        
        if self.config['enable_hyperparameter_tuning'] and param_grid:
            # Grid search with cross-validation
            cv = StratifiedKFold(n_splits=self.config['cross_validation_folds'], 
                               shuffle=True, random_state=self.config['random_state'])
            
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring='f1_macro',
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        else:
            # Train with default parameters
            best_model = model
            best_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, 
                                  cv=self.config['cross_validation_folds'], 
                                  scoring='f1_macro')
        
        results = {
            'model': best_model,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'training_time': time.time()
        }
        
        logger.info(f"{name} - Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, CV: {np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")
        
        return results
    
    def create_ensemble_model(self, X_train, y_train, X_val, y_val):
        """Create an ensemble model from the best individual models"""
        if not self.config['enable_ensemble'] or len(self.models) < 2:
            return None
        
        logger.info("Creating ensemble model...")
        
        # Select top 3 models based on validation F1 score
        sorted_models = sorted(self.models.items(), 
                             key=lambda x: x[1]['val_f1'], reverse=True)
        top_models = sorted_models[:3]
        
        # Create voting classifier
        estimators = [(name, results['model']) for name, results in top_models]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        val_pred = ensemble.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        ensemble_results = {
            'model': ensemble,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1,
            'cv_mean': 0,  # Not computed for ensemble
            'cv_std': 0,
            'training_time': time.time(),
            'component_models': [name for name, _ in top_models]
        }
        
        logger.info(f"Ensemble - Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        logger.info(f"Ensemble components: {[name for name, _ in top_models]}")
        
        return ensemble_results
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive model evaluation"""
        logger.info(f"Evaluating {model_name} on test set...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Per-class metrics
        class_report = classification_report(y_test, y_pred, 
                                           target_names=self.label_encoder.classes_,
                                           output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"{model_name} Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        return evaluation_results
    
    def plot_confusion_matrix(self, cm, model_name, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, model_name, save_path=None, top_n=20):
        """Plot feature importance for tree-based models"""
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature importance")
            return
        
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.bar(range(top_n), importance[indices])
        plt.xticks(range(top_n), [f'Feature_{indices[i]}' for i in range(top_n)], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, model, X_train, y_train, model_name, save_path=None):
        """Plot learning curves"""
        from sklearn.model_selection import learning_curve
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, train_sizes=train_sizes, 
            cv=5, scoring='f1_macro', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.title(f'Learning Curves - {model_name}')
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_and_results(self, model, evaluation_results, model_name):
        """Save the trained model and results"""
        results_dir = Path(self.config['results_dir'])
        
        # Save model
        model_path = results_dir / f"{model_name.lower()}_model.pkl"
        
        # Create model package with preprocessing components
        model_package = {
            'model': model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'training_date': datetime.now().isoformat(),
            'evaluation_results': evaluation_results
        }
        
        joblib.dump(model_package, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save detailed results
        results_path = results_dir / f"{model_name.lower()}_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = evaluation_results.copy()
        if 'confusion_matrix' in json_results:
            json_results['confusion_matrix'] = json_results['confusion_matrix'].tolist()
        if 'predictions' in json_results:
            json_results['predictions'] = json_results['predictions'].tolist()
        if 'probabilities' in json_results and json_results['probabilities'] is not None:
            json_results['probabilities'] = json_results['probabilities'].tolist()
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        return model_path
    
    def train_and_evaluate(self):
        """Main training and evaluation pipeline"""
        logger.info("Starting enhanced model training pipeline...")
        start_time = time.time()
        
        try:
            # Load and prepare data
            X, y, feature_cols = self.load_and_prepare_data()
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            
            # Preprocess features
            X_train, X_val, X_test = self.preprocess_features(X_train, X_val, X_test)
            
            # Feature selection
            X_train, X_val, X_test = self.feature_selection(X_train, y_train, X_val, X_test)
            
            # Dimensionality reduction
            X_train, X_val, X_test = self.dimensionality_reduction(X_train, X_val, X_test)
            
            # Train individual models
            models_config = self.define_models()
            
            for name, config in models_config.items():
                try:
                    results = self.train_single_model(name, config, X_train, y_train, X_val, y_val)
                    self.models[name] = results
                except Exception as e:
                    logger.error(f"Failed to train {name}: {str(e)}")
                    continue
            
            # Create ensemble model
            ensemble_results = self.create_ensemble_model(X_train, y_train, X_val, y_val)
            if ensemble_results:
                self.models['Ensemble'] = ensemble_results
            
            # Select best model based on validation F1 score
            best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['val_f1'])
            self.best_model = self.models[best_model_name]['model']
            
            logger.info(f"Best model: {best_model_name} (F1: {self.models[best_model_name]['val_f1']:.4f})")
            
            # Evaluate best model on test set
            evaluation_results = self.evaluate_model(self.best_model, X_test, y_test, best_model_name)
            
            # Generate visualizations
            results_dir = Path(self.config['results_dir'])
            
            # Confusion matrix
            cm_path = results_dir / f"{best_model_name.lower()}_confusion_matrix.png"
            self.plot_confusion_matrix(evaluation_results['confusion_matrix'], 
                                     best_model_name, str(cm_path))
            
            # Feature importance (if available)
            if hasattr(self.best_model, 'feature_importances_'):
                fi_path = results_dir / f"{best_model_name.lower()}_feature_importance.png"
                self.plot_feature_importance(self.best_model, feature_cols, 
                                           best_model_name, str(fi_path))
            
            # Learning curves
            lc_path = results_dir / f"{best_model_name.lower()}_learning_curves.png"
            self.plot_learning_curves(self.best_model, X_train, y_train, 
                                    best_model_name, str(lc_path))
            
            # Save model and results
            model_path = self.save_model_and_results(self.best_model, evaluation_results, best_model_name)
            
            # Copy to main model path for compatibility
            import shutil
            main_model_path = Path(self.config['model_path'])
            shutil.copy2(model_path, main_model_path)
            logger.info(f"Model also saved to {main_model_path}")
            
            # Training summary
            total_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total training time: {total_time:.2f} seconds")
            logger.info(f"Best model: {best_model_name}")
            logger.info(f"Test accuracy: {evaluation_results['accuracy']:.4f}")
            logger.info(f"Test F1-score: {evaluation_results['f1']:.4f}")
            logger.info(f"Model saved to: {main_model_path}")
            
            # Model comparison
            logger.info("\nModel Comparison (Validation F1):")
            for name, results in sorted(self.models.items(), 
                                      key=lambda x: x[1]['val_f1'], reverse=True):
                logger.info(f"  {name}: {results['val_f1']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_model_for_inference(self, model_path=None):
        """Load a trained model for inference"""
        if model_path is None:
            model_path = self.config['model_path']
        
        try:
            model_package = joblib.load(model_path)
            
            # Extract components
            if isinstance(model_package, dict):
                # New format with preprocessing pipeline
                self.best_model = model_package['model']
                self.scaler = model_package.get('scaler')
                self.feature_selector = model_package.get('feature_selector')
                self.pca = model_package.get('pca')
                self.label_encoder = model_package.get('label_encoder')
                logger.info("Loaded model package with preprocessing pipeline")
            else:
                # Legacy format - just the model
                self.best_model = model_package
                logger.warning("Loaded legacy model format without preprocessing pipeline")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def predict(self, X):
        """Make predictions with the loaded model"""
        if self.best_model is None:
            raise ValueError("No model loaded. Call load_model_for_inference() first.")
        
        # Apply preprocessing pipeline
        if self.scaler:
            X = self.scaler.transform(X)
        
        if self.feature_selector:
            X = self.feature_selector.transform(X)
        
        if self.pca:
            X = self.pca.transform(X)
        
        # Make predictions
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X) if hasattr(self.best_model, 'predict_proba') else None
        
        # Decode labels
        if self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions, probabilities

def main():
    parser = argparse.ArgumentParser(description='Enhanced Sign Language Model Trainer')
    parser.add_argument('--csv_path', default='processed_landmarks.csv',
                       help='Path to processed CSV file')
    parser.add_argument('--model_path', default='sign_language_model.pkl',
                       help='Path to save the trained model')
    parser.add_argument('--results_dir', default='training_results',
                       help='Directory to save training results')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (0-1)')
    parser.add_argument('--no_ensemble', action='store_true',
                       help='Disable ensemble modeling')
    parser.add_argument('--no_tuning', action='store_true',
                       help='Disable hyperparameter tuning')
    parser.add_argument('--enable_pca', action='store_true',
                       help='Enable PCA dimensionality reduction')
    parser.add_argument('--pca_components', type=int, default=50,
                       help='Number of PCA components')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Cross-validation folds')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'csv_path': args.csv_path,
        'model_path': args.model_path,
        'results_dir': args.results_dir,
        'test_size': args.test_size,
        'validation_size': 0.1,
        'random_state': 42,
        'enable_feature_selection': True,
        'enable_dimensionality_reduction': args.enable_pca,
        'enable_ensemble': not args.no_ensemble,
        'enable_hyperparameter_tuning': not args.no_tuning,
        'cross_validation_folds': args.cv_folds,
        'max_features_pca': args.pca_components,
        'scale_features': True
    }
    
    # Create trainer and run
    trainer = EnhancedModelTrainer(config)
    
    success = trainer.train_and_evaluate()
    
    if success:
        logger.info("✅ Model training completed successfully!")
        return 0
    else:
        logger.error("❌ Model training failed!")
        return 1

if __name__ == "__main__":
    exit(main())