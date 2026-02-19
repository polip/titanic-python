import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from datetime import datetime

# ✅ Add MLflow import (optional)
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    print("✅ MLflow available for experiment tracking")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not installed - install with: pip install mlflow")

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline using ColumnTransformer.
    """
    # Numeric features for imputation and scaling
    numeric_features = ['age', 'fare', 'sibsp', 'parch']
    
    # Categorical features for encoding
    categorical_features = ['pclass', 'sex', 'embarked', 'title']
    
    # Numeric preprocessing: KNN imputation + scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing: handle missing values + encoding
    categorical_transformer = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def train_model(data_path='data/train_clean.pkl', model_path='models/titanic_model_ct.pkl'):
    """
    Train Random Forest model using the ColumnTransformer pipeline with optional MLflow tracking.
    """
    # ✅ Setup MLflow if available and enabled
    use_mlflow = MLFLOW_AVAILABLE

    if use_mlflow:
        # Setup MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("titanic-survival")
        mlflow.start_run()
        print("🔬 MLflow tracking enabled")
    
    try:
        # Load cleaned data
        train_clean = pd.read_pickle(data_path)
        print(f"📥 Data loaded: {train_clean.shape}")
        
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        if use_mlflow:
            mlflow.end_run()
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        if use_mlflow:
            mlflow.end_run()
        return None, None
    
    # Handle the 'pclass' mapping and 'title' extraction manually before passing to the pipeline
    train_clean['pclass'] = train_clean['pclass'].map({1: '1st', 2: '2nd', 3: '3rd'}).fillna('unknown')
    
    print("Training data shape:", train_clean.shape)
    print("Columns:", list(train_clean.columns))
    
    # Prepare features and target
    X = train_clean.drop(['survived', 'passengerid'], axis=1)
    y = train_clean['survived']
    
    # ✅ Log data info to MLflow
    if use_mlflow:
        mlflow.log_params({
            "data_path": str(data_path),
            "n_samples": len(train_clean),
            "n_features": len(X.columns)
        })
    
    # Create the full pipeline with preprocessor and classifier
    model_params = {
        'n_estimators': 500,
        'random_state': 42,
        'n_jobs': -1
    }
    
    pipeline = Pipeline([
        ('preprocessor', create_preprocessing_pipeline()),
        ('classifier', RandomForestClassifier(**model_params))
    ])
    
    # ✅ Log model parameters to MLflow
    if use_mlflow:
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            **model_params,
            "preprocessing": "ColumnTransformer"
        })
    

   
    # Cross-validation accuracy evaluation
    cv_scores_acc = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    cv_acc_mean = cv_scores_acc.mean()
    cv_acc_sd = cv_scores_acc.std()
    print(f"Cross-validation accuracy: {cv_acc_mean:.3f} (+/- {cv_acc_sd * 2:.3f})")

    # Cross-validation recall evaluation
    cv_scores_recall = cross_val_score(pipeline, X, y, cv=5, scoring='recall')
    cv_recall_mean = cv_scores_recall.mean()
    cv_recall_std = cv_scores_recall.std()    
    print(f"Cross-validation recall score: {cv_recall_mean:.3f} (+/- {cv_recall_std * 2:.3f})")
    

    # Final training
    print("Processing data and training model...")
    pipeline.fit(X, y) 
    
    # Training accuracy
    train_accuracy = pipeline.score(X, y)
    print(f"Training accuracy: {train_accuracy:.3f}")

    
    # ✅ Log metrics to MLflow
    if use_mlflow:
        mlflow.log_metrics({
            "cv_acc_mean": cv_acc_mean,
            "cv_acc_std": cv_acc_sd,
            "cv_recall_mean": cv_recall_mean,
            "cv_recall_std": cv_recall_std,
            "train_accuracy": train_accuracy,
            "n_estimators": model_params['n_estimators']
        })
    
    # Feature importance
    # Get feature names after preprocessing
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importance:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance_ct.png', dpi=300, bbox_inches='tight')
    
    # ✅ Log plot to MLflow
    if use_mlflow:
        mlflow.log_artifact('feature_importance_ct.png')
        
        # Also log feature importance as CSV
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
    
    # Show plot (skip if using MLflow to avoid blocking)
    if not use_mlflow:
        plt.show()
    else:
        plt.close()  # Close to avoid memory issues
    
    # Save model and feature names
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    joblib.dump(feature_names, model_path.replace('.pkl', '_features.pkl'))
    
    print(f"Model saved to {model_path}")
    
    # ✅ Log model to MLflow
    if use_mlflow:
        if cv_acc_mean > 0.9:  # Log only if accuracy is reasonable
            mlflow.sklearn.log_model(
                pipeline, 
                "model",
                registered_model_name="titanic-survival-model"
         )

     # ✅ End MLflow run
    if use_mlflow:
        mlflow.end_run()
        print("✅ MLflow run completed")
    
    return pipeline, feature_importance


def predict_test_data(model_path='models/titanic_model_ct.pkl', 
                     test_data_path='data/test.csv',
                     output_path='data/submission_python_ct.csv'):
    """
    Make predictions on test data using the ColumnTransformer pipeline.
    """
    
    try:
        # Load model
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: The model file '{model_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None
        
    try:
        # Load and preprocess test data
        test = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print(f"Error: The test data file '{test_data_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the test data: {e}")
        return None
        
    test.columns = test.columns.str.lower().str.replace(' ', '_')
    
    # Apply same feature engineering as training data
    test['pclass'] = test['pclass'].map({1: '1st', 2: '2nd', 3: '3rd'}).fillna('unknown')
    
    def extract_title(name):
        match = re.search(r',\s*(.*?)\s*\.', name)
        if match:
            return match.group(1).strip()
        return None
    
    test['title'] = test['name'].apply(extract_title).fillna('unknown')

    # Prepare features, the pipeline will handle all other preprocessing
    X_test = test.drop(['passengerid', 'name', 'ticket', 'cabin'], axis=1)

    # Make predictions
    predictions_class = model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'PassengerId': test['passengerid'],
        'Survived': predictions_class.astype(int)
    })
    
    # Save submission
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"Submission file saved to {output_path}")
    print(f"Predictions summary:")
    print(submission['Survived'].value_counts())
    
    return submission

if __name__ == "__main__":
    
    # Train model
    print("Training model...")
    model, importance = train_model()
    
    # Make test predictions
    if model is not None:
        print("\nMaking test predictions...")
        submission = predict_test_data()
    
    print("\nTraining completed successfully!")