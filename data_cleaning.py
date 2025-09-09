"""
Data cleaning and preprocessing for Titanic dataset
Python translation of cleaning.R
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def clean_titanic_data(input_path='data/train.csv', output_path='data/train_clean.pkl'):
    """
    Clean and preprocess Titanic training data
    
    Args:
        input_path (str): Path to raw training data
        output_path (str): Path to save cleaned data
    
    Returns:
        pd.DataFrame: Cleaned training data
    """
    
    # Load data
    train = pd.read_csv(input_path)
    
    # Clean column names (equivalent to janitor::clean_names())
    train.columns = train.columns.str.lower().str.replace(' ', '_')
    
    # Convert survived to categorical with proper ordering (0 first, then 1)
    train['survived'] = pd.Categorical(train['survived'], categories=[0, 1], ordered=True)
    
    # Convert pclass to categorical with labels
    train['pclass'] = pd.Categorical(
        train['pclass'].map({1: '1st', 2: '2nd', 3: '3rd'}),
        categories=['1st', '2nd', '3rd'],
        ordered=True
    )
    
    # Convert sex to categorical
    train['sex'] = pd.Categorical(train['sex'])
    
    # Convert embarked to categorical
    train['embarked'] = pd.Categorical(train['embarked'])
    
    # Extract title from name using regex
    def extract_title(name):
        """Extract title from passenger name"""
        match = re.search(r',\s*(.*?)\s*\.', name)
        if match:
            title = match.group(1).strip()
            return title
        return None
    
    train['title'] = train['name'].apply(extract_title)
    train['title'] = pd.Categorical(train['title'])
    
    # Remove unnecessary columns
    columns_to_drop = ['name', 'ticket', 'cabin']
    train_clean = train.drop(columns=columns_to_drop)
    
    # Save cleaned data
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    train_clean.to_pickle(output_path)
    
    print(f"Cleaned data saved to {output_path}")
    print(f"Shape: {train_clean.shape}")
    print(f"Columns: {list(train_clean.columns)}")
    
    return train_clean

if __name__ == "__main__":
    # Clean the data
    cleaned_data = clean_titanic_data()
    
    # Display basic info
    print("\nData Info:")
    print(cleaned_data.info())
    
    print("\nSurvival counts:")
    print(cleaned_data['survived'].value_counts())
    
    print("\nFirst few rows:")
    print(cleaned_data.head())