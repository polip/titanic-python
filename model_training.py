import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re

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
    Train Random Forest model using the ColumnTransformer pipeline.
    """
    try:
        # Load cleaned data
        train_clean = pd.read_pickle(data_path)
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None, None
    
    # Handle the 'pclass' mapping and 'title' extraction manually before passing to the pipeline
    train_clean['pclass'] = train_clean['pclass'].map({1: '1st', 2: '2nd', 3: '3rd'}).fillna('unknown')
    
    
    print("Training data shape:", train_clean.shape)
    print("Columns:", list(train_clean.columns))
    
    # Prepare features and target
    X = train_clean.drop(['survived', 'passengerid'], axis=1)
    y = train_clean['survived']
    
    # Create the full pipeline with preprocessor and classifier
    pipeline = Pipeline([
        ('preprocessor', create_preprocessing_pipeline()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Processing data and training model...")
    pipeline.fit(X, y)
    
    # Cross-validation evaluation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Feature importance
    # Get feature names after preprocessing
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance_ct.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model and feature names
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    joblib.dump(feature_names, model_path.replace('.pkl', '_features.pkl'))
    
    print(f"Model saved to {model_path}")
    
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
