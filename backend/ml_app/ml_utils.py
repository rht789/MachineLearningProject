import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, silhouette_score
)
from typing import Dict, Any, Tuple, List
import logging
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate the input dataset"""
        try:
            # Check if dataframe is empty
            if df.empty:
                return False, "Dataset is empty"
            
            # Check for missing values
            missing_percent = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            if missing_percent > 30:
                return False, f"Dataset contains too many missing values ({missing_percent:.2f}%)"
            
            # Check for minimum number of rows
            if len(df) < 10:
                return False, "Dataset must contain at least 10 rows"
            
            return True, "Validation successful"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = None
        
    def preprocess_data(self, df: pd.DataFrame, target_column: str = None, objective: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the dataset"""
        try:
            # Make a copy of the dataframe to avoid modifying the original
            df = df.copy()
            
            # Validate target column exists
            if target_column and target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Handle missing values first
            df = self._handle_missing_values(df)
            
            # Separate features and target
            if target_column:
                y = df[target_column].copy()
                X = df.drop(columns=[target_column])
            else:
                X = df.copy()
                y = None
            
            # Handle categorical variables in features
            X = self._encode_categorical_variables(X)
            
            # Handle target variable based on objective
            if y is not None:
                if objective == 'classification':
                    y = self._encode_target_variable(y)
                elif objective == 'regression':
                    try:
                        y = y.astype(float)
                    except ValueError:
                        raise ValueError("Target column must be numeric for regression")
            
            # Scale numerical features
            X = self._scale_features(X)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise ValueError(f"Error preprocessing data: {str(e)}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            # Fill numeric columns with median
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_columns.empty:
                for col in numeric_columns:
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical columns with mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            if not categorical_columns.empty:
                for col in categorical_columns:
                    if df[col].isnull().any():
                        mode_value = df[col].mode()
                        if not mode_value.empty:
                            df[col] = df[col].fillna(mode_value[0])
            
            return df
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        try:
            df = df.copy()
            categorical_columns = df.select_dtypes(include=['object']).columns
            
            for column in categorical_columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                
                # Handle missing values before encoding
                if df[column].isnull().any():
                    df[column] = df[column].fillna('missing')
                
                df[column] = self.label_encoders[column].fit_transform(df[column].astype(str))
            
            return df
        except Exception as e:
            logger.error(f"Error encoding categorical variables: {str(e)}")
            raise
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        try:
            # Only scale numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_columns.empty:
                scaled_data = self.scaler.fit_transform(df[numeric_columns])
                df[numeric_columns] = scaled_data
            return df
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise
    
    def _encode_target_variable(self, y: pd.Series) -> pd.Series:
        """Encode categorical target variable"""
        try:
            if self.target_encoder is None:
                self.target_encoder = LabelEncoder()
            
            # Handle missing values if any
            if y.isnull().any():
                y = y.fillna('missing')
            
            return pd.Series(
                self.target_encoder.fit_transform(y.astype(str)),
                index=y.index
            )
        except Exception as e:
            logger.error(f"Error encoding target variable: {str(e)}")
            raise
    
    def get_target_classes(self) -> List[str]:
        """Get original target classes"""
        return list(self.target_encoder.classes_) if self.target_encoder else []

class DataBalancer:
    """Handle class imbalance in classification tasks"""
    def __init__(self, min_samples: int = 2):
        self.min_samples = min_samples
        self.strategy = None
        
    def balance_dataset(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance dataset using appropriate strategy"""
        try:
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            
            # If severely imbalanced, use appropriate resampling strategy
            if min_class_count < self.min_samples:
                # For extremely small datasets (less than 2 samples in minority class)
                if min_class_count < 2:
                    # Simple duplication for very small classes
                    X_resampled = X.copy()
                    y_resampled = y.copy()
                    
                    # Duplicate minority class samples
                    for class_label in class_counts[class_counts < 2].index:
                        minority_mask = y == class_label
                        minority_X = X[minority_mask]
                        minority_y = y[minority_mask]
                        
                        # Duplicate the single sample
                        X_resampled = pd.concat([X_resampled, minority_X])
                        y_resampled = pd.concat([y_resampled, minority_y])
                    
                    self.strategy = 'simple_duplication'
                    return X_resampled, y_resampled
                
                else:
                    # Use SMOTE for datasets with at least 2 samples per class
                    target_samples = max(self.min_samples, int(max_class_count * 0.5))
                    
                    # Configure SMOTE with appropriate k_neighbors
                    k_neighbors = min(min_class_count - 1, 5)
                    k_neighbors = max(1, k_neighbors)  # Ensure k_neighbors is at least 1
                    
                    oversample = SMOTE(
                        sampling_strategy={
                            label: target_samples 
                            for label in class_counts[class_counts < target_samples].index
                        },
                        random_state=42,
                        k_neighbors=k_neighbors
                    )
                    
                    X_resampled, y_resampled = oversample.fit_resample(X, y)
                    self.strategy = 'SMOTE'
                    return X_resampled, y_resampled
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error in balancing dataset: {str(e)}")
            # If balancing fails, return original data with warning
            logger.warning("Data balancing failed, proceeding with original data")
            return X, y

    def _validate_resampled_data(self, y_resampled: pd.Series) -> bool:
        """Validate that resampling produced valid results"""
        class_counts = y_resampled.value_counts()
        return class_counts.min() >= self.min_samples

class MLModel:
    ALGORITHMS = {
        'regression': {
            'linear_regression': LinearRegression,
            'knn_regressor': KNeighborsRegressor
        },
        'classification': {
            'logistic_regression': LogisticRegression,
            'knn_classifier': KNeighborsClassifier,
            'naive_bayes': GaussianNB
        },
        'clustering': {
            'kmeans': KMeans,
            'hierarchical': AgglomerativeClustering
        }
    }

    def __init__(self, objective: str):
        if objective not in self.ALGORITHMS:
            raise ValueError(f"Unsupported objective: {objective}")
        
        self.objective = objective
        self.models = self._initialize_models()
        self.balancer = DataBalancer() if objective == 'classification' else None

    def _initialize_models(self):
        """Initialize all models for the given objective"""
        models = {}
        for name, model_class in self.ALGORITHMS[self.objective].items():
            if model_class == KNeighborsRegressor or model_class == KNeighborsClassifier:
                models[name] = model_class(n_neighbors=5)
            elif model_class == LogisticRegression:
                models[name] = model_class(random_state=42)
            elif model_class == KMeans:
                models[name] = model_class(n_clusters=3, random_state=42)
            elif model_class == AgglomerativeClustering:
                models[name] = model_class(n_clusters=3)
            else:
                models[name] = model_class()
        return models
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """Train all models and evaluate their performance"""
        try:
            if self.objective in ['classification', 'regression']:
                return self._supervised_learning(X, y)
            else:
                return self._clustering(X)
        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
            raise

    def _supervised_learning(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Handle supervised learning tasks"""
        try:
            # Initialize variables
            original_distribution = None
            new_distribution = None
            y_balanced = None
            
            # Handle class imbalance for classification
            if self.objective == 'classification':
                class_counts = y.value_counts()
                original_distribution = dict(class_counts)
                
                if class_counts.min() < 2:
                    X_balanced, y_balanced = self.balancer.balance_dataset(X, y)
                    new_distribution = dict(pd.Series(y_balanced).value_counts())
                    
                    if self.balancer.strategy:
                        X, y = X_balanced, y_balanced
                        logger.info(f"Applied {self.balancer.strategy} for data balancing")
                else:
                    new_distribution = original_distribution

            # Split data
            test_size = 0.2 if len(y) >= 10 else 0.3
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=42,
                    stratify=y if self.objective == 'classification' else None
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=42
                )

            # Train and evaluate all models
            results = {}
            for name, model in self.models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred)
                
                # Add feature importance if available
                if hasattr(model, 'feature_importances_'):
                    metrics['feature_importance'] = dict(sorted(
                        zip(X.columns, model.feature_importances_),
                        key=lambda x: x[1],
                        reverse=True
                    ))
                elif hasattr(model, 'coef_'):
                    metrics['feature_importance'] = dict(sorted(
                        zip(X.columns, abs(model.coef_)),
                        key=lambda x: x[1],
                        reverse=True
                    ))
                
                results[name] = metrics

            # Add distribution info
            if self.objective == 'classification':
                results['data_distribution'] = {
                    'original': original_distribution,
                    'after_balancing': new_distribution,
                    'train_set': dict(pd.Series(y_train).value_counts()),
                    'test_set': dict(pd.Series(y_test).value_counts())
                }
                results['balancing_info'] = {
                    'strategy_used': self.balancer.strategy or 'none',
                    'samples_before': len(y),
                    'samples_after': len(y_balanced) if y_balanced is not None else len(y)
                }
            else:
                results['data_distribution'] = {
                    'train_samples': len(y_train),
                    'test_samples': len(y_test)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in supervised learning: {str(e)}")
            raise ValueError(f"Analysis failed: {str(e)}")

    def _clustering(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Handle clustering tasks"""
        results = {}
        
        for name, model in self.models.items():
            # Fit the clustering model
            cluster_labels = model.fit_predict(X)
            
            # Calculate clustering metrics
            metrics = {
                'silhouette_score': float(silhouette_score(X, cluster_labels)),
                'n_clusters': getattr(model, 'n_clusters', len(np.unique(cluster_labels))),
                'cluster_sizes': dict(pd.Series(cluster_labels).value_counts().to_dict())
            }
            
            # Add inertia if available (K-means specific)
            if hasattr(model, 'inertia_'):
                metrics['inertia'] = float(model.inertia_)
            
            results[name] = metrics
        
        return results

    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate metrics based on the objective"""
        try:
            if self.objective == 'classification':
                # Handle binary and multiclass cases
                average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
                return {
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
                    'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
                }
            else:  # regression
                mse = float(mean_squared_error(y_true, y_pred))
                return {
                    'r2_score': float(r2_score(y_true, y_pred)),
                    'mse': mse,
                    'rmse': float(np.sqrt(mse))
                }
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

def analyze_dataset(file_path: str, objective: str, target_column: str = None) -> Dict[str, Any]:
    """Main function to analyze the dataset"""
    try:
        # Read the dataset
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # For classification, check class distribution
        if objective == 'classification' and target_column:
            unique_classes = df[target_column].nunique()
            if unique_classes < 2:
                raise ValueError("Classification requires at least 2 classes in target column")
            
            class_counts = df[target_column].value_counts()
            if class_counts.min() < 2:
                logger.warning("Small class sizes detected. Will apply data balancing techniques.")
        
        # Validate dataset
        validator = DataValidator()
        is_valid, message = validator.validate_dataset(df)
        if not is_valid:
            raise ValueError(message)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_data(df, target_column, objective)
        
        # Initialize model and get results
        model = MLModel(objective)
        metrics = model.train_and_evaluate(X, y)
        
        # Convert numpy types to Python native types
        if objective == 'classification':
            target_classes = [str(c) for c in preprocessor.get_target_classes()]
            class_distribution = y.value_counts().to_dict()
            class_distribution = {str(k): int(v) for k, v in class_distribution.items()}
        else:
            target_classes = None
            class_distribution = None

        # Prepare final results with serializable types
        results = {
            'algorithm_results': metrics,
            'data_shape': {
                'rows': int(len(df)),
                'columns': int(len(df.columns))
            },
            'preprocessing_info': {
                'missing_values_handled': True,
                'categorical_columns': list(preprocessor.label_encoders.keys()),
                'scaled_features': True,
                'target_type': 'categorical' if objective == 'classification' else 'numeric',
                'small_dataset_handling': bool(len(df) < 20)
            }
        }

        # Add classification-specific information
        if objective == 'classification':
            results['target_classes'] = target_classes
            results['class_distribution'] = class_distribution

        return results
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise ValueError(f"Error analyzing dataset: {str(e)}")
