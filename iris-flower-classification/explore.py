import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def explore_data(df):
    print("=== Shape ===")
    print(df.shape)

    print("\n=== First 5 Rows ===")
    print(df.head())

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== Class Distribution ===")
    print(df["species"].value_counts())

    print("\n=== Basic Statistics ===")
    print(df.describe())