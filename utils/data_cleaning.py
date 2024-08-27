import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(df):
    df = df[['Pclass', 'Sex', 'Age', 'Survived']].copy()
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'].fillna(df['Age'].median(), inplace=True)
    return df

def split_data(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)
