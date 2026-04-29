import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess(df):
    X = df.drop(columns="species")
    y = df["species"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model