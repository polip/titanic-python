"""
Shared utilities for Titanic prediction project
Contains feature encoding mappings and preprocessing functions
"""

import pandas as pd

# Feature encoding mappings
PCLASS_MAP = {'1st': 0, '2nd': 1, '3rd': 2}
SEX_MAP = {'male': 1, 'female': 0}
EMBARKED_MAP = {'S': 2, 'C': 0, 'Q': 1}
TITLE_MAP = {
    'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3,
    'Dr': 4, 'Rev': 5, 'Col': 6, 'Mlle': 7, 'Major': 8,
    'Other': 9
}

def encode_features(pclass, sex, age, sibsp, parch, fare, embarked, title):
    """
    Encode passenger features for model prediction

    Args:
        pclass: Passenger class ('1st', '2nd', '3rd')
        sex: Sex ('male', 'female')
        age: Age in years
        sibsp: Number of siblings/spouses
        parch: Number of parents/children
        fare: Fare paid
        embarked: Port of embarkation ('S', 'C', 'Q')
        title: Title ('Mr', 'Mrs', 'Miss', etc.)

    Returns:
        list: Encoded feature vector
    """
    features = [
        PCLASS_MAP.get(pclass, 2),
        SEX_MAP.get(sex, 0),
        age,
        sibsp,
        parch,
        fare,
        EMBARKED_MAP.get(embarked, 2),
        TITLE_MAP.get(title, 9)
    ]
    return features

def encode_dataframe(df):
    """
    Encode a dataframe with passenger data

    Args:
        df: DataFrame with columns: pclass, sex, age, sibsp, parch, fare, embarked, title

    Returns:
        DataFrame: Encoded dataframe
    """
    encoded_df = df.copy()
    encoded_df['pclass'] = encoded_df['pclass'].map(PCLASS_MAP)
    encoded_df['sex'] = encoded_df['sex'].map(SEX_MAP)
    encoded_df['embarked'] = encoded_df['embarked'].map(EMBARKED_MAP)
    encoded_df['title'] = encoded_df['title'].map(TITLE_MAP)
    return encoded_df
