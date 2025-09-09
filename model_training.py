"""
Model training for Titanic survival prediction
Python translation of titanic_training.R
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def create_preprocessing_pipeline():
    """
    Create preprocessing pipeline equivalent to R recipe
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
        ('encoder', LabelEncoder())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop passenger_id
    )
    
    return preprocessor

def prepare_features(df):
    """
    Prepare features for modeling
    """
    # Handle categorical encoding manually due to ColumnTransformer limitations with pandas categoricals
    df_processed = df.copy()
    
    # Convert categorical columns to strings first, then use LabelEncoder
    categorical_columns = ['pclass', 'sex', 'embarked', 'title']
    
    for col in categorical_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
            le = LabelEncoder()
            # Handle missing values
            df_processed[col] = df_processed[col].fillna('unknown')
            df_processed[col] = le.fit_transform(df_processed[col])
    
    # Handle numeric missing values with KNN imputation
    numeric_columns = ['age', 'fare', 'sibsp', 'parch']
    
    if any(col in df_processed.columns for col in numeric_columns):
        imputer = KNNImputer(n_neighbors=5)
        existing_numeric = [col for col in numeric_columns if col in df_processed.columns]
        df_processed[existing_numeric] = imputer.fit_transform(df_processed[existing_numeric])
    
    return df_processed

def train_model(data_path='data/train_clean.pkl', model_path='models/titanic_model.pkl'):
    """
    Train Random Forest model equivalent to R workflow
    """
    
    # Load cleaned data
    train_clean = pd.read_pickle(data_path)
    
    print("Training data shape:", train_clean.shape)
    print("Columns:", list(train_clean.columns))
    
    # Prepare features
    X = train_clean.drop(['survived'], axis=1)
    y = train_clean['survived']
    
    # Convert y to numeric if it's categorical
    if hasattr(y, 'cat'):
        y = y.cat.codes
    
    # Prepare features
    X_processed = prepare_features(X)
    
    # Remove passenger_id if present
    if 'passengerid' in X_processed.columns:
        X_processed = X_processed.drop(['passengerid'], axis=1)
    if 'passenger_id' in X_processed.columns:
        X_processed = X_processed.drop(['passenger_id'], axis=1)
    
    print("Processed features shape:", X_processed.shape)
    print("Features:", list(X_processed.columns))
    
    # Create Random Forest model (equivalent to ranger in R)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', rf_model)
    ])
    
    # Train model
    pipeline.fit(X_processed, y)
    
    # Cross-validation evaluation
    cv_scores = cross_val_score(pipeline, X_processed, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('python_version/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    
    # Save feature names for later use
    feature_names = list(X_processed.columns)
    joblib.dump(feature_names, model_path.replace('.pkl', '_features.pkl'))
    
    print(f"Model saved to {model_path}")
    
    return pipeline, feature_importance

def predict_test_data(model_path='models/titanic_model.pkl', 
                     test_data_path='data/test.csv',
                     output_path='data/submission_python.csv'):
    """
    Make predictions on test data and create submission file
    """
    
    # Load model and feature names
    model = joblib.load(model_path)
    feature_names = joblib.load(model_path.replace('.pkl', '_features.pkl'))
    
    # Load and preprocess test data
    test = pd.read_csv(test_data_path)
    test.columns = test.columns.str.lower().str.replace(' ', '_')
    
    # Apply same preprocessing as training data
    test['pclass'] = pd.Categorical(
        test['pclass'].map({1: '1st', 2: '2nd', 3: '3rd'}),
        categories=['1st', '2nd', '3rd'],
        ordered=True
    )
    
    # Extract title
    def extract_title(name):
        match = re.search(r',\s*(.*?)\s*\.', name)
        if match:
            return match.group(1).strip()
        return None
    
    test['title'] = test['name'].apply(extract_title)
    test['title'] = pd.Categorical(test['title'])
    
    # Process features
    X_test = prepare_features(test)
    
    # Keep only features that were used in training
    X_test = X_test[feature_names]
    
    # Make predictions
    predictions_proba = model.predict_proba(X_test)
    predictions_class = model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'PassengerId': test['passengerid'],
        'Survived': predictions_class
    })
    
    # Save submission
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"Submission file saved to {output_path}")
    print(f"Predictions summary:")
    print(submission['Survived'].value_counts())
    
    return submission, predictions_proba

if __name__ == "__main__":
    import re
    
    # Train model
    print("Training model...")
    model, importance = train_model()
    
    # Make test predictions
    print("\nMaking test predictions...")
    submission, probabilities = predict_test_data()
    
    print("\nTraining completed successfully!")