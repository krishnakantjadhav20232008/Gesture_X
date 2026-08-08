import pandas as pd
import numpy as np

# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_csv(
    r"D:\AI_Sign_Gesture_System\outputs\gesturex_dataset.csv"
)
print(df.head())

# ==========================================
# BASIC EDA
# ==========================================

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nClass Distribution")
print(df["label"].value_counts())