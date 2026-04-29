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

def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)

    print("=== Accuracy ===")
    print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved as confusion_matrix.png")

    return y_pred