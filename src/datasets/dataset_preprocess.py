import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

CSV_FILE = r"D:\AI_Sign_Gesture_System\outputs\gesturex_dataset.csv"

df = pd.read_csv(CSV_FILE)

print("\n===================================")
print("DATASET LOADED")
print("===================================")

print("Dataset Shape:", df.shape)


# ==========================================================
# 2. BASIC EDA
# ==========================================================

print("\n===================================")
print("BASIC EDA")
print("===================================")

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum().sum())


# ==========================================================
# 3. DUPLICATE CHECK
# ==========================================================

print("\n===================================")
print("DUPLICATE CHECK")
print("===================================")

duplicate_count = df.duplicated().sum()

print("Duplicate Rows:", duplicate_count)


# Remove duplicate rows
df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)


# ==========================================================
# 4. CHECK MISSING VALUES AGAIN
# ==========================================================

print("\n===================================")
print("MISSING VALUE CHECK")
print("===================================")

missing_values = df.isnull().sum().sum()

print("Total Missing Values:", missing_values)


# ==========================================================
# 5. CLASS DISTRIBUTION
# ==========================================================

print("\n===================================")
print("CLASS DISTRIBUTION")
print("===================================")

print(df["label"].value_counts().sort_index())


# ==========================================================
# 6. SEPARATE FEATURES AND TARGET
# ==========================================================

print("\n===================================")
print("FEATURE / TARGET SEPARATION")
print("===================================")

# X = Landmark features
X = df.drop("label", axis=1)

# y = Gesture/sign labels
y = df["label"]

print("Features Shape:", X.shape)
print("Target Shape :", y.shape)


# ==========================================================
# 7. LABEL ENCODING
# ==========================================================

print("\n===================================")
print("LABEL ENCODING")
print("===================================")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("Label Encoding Completed")

print("\nEncoded Classes:")
print(label_encoder.classes_)

print("\nNumber of Classes:",
      len(label_encoder.classes_))


# ==========================================================
# 8. TRAIN-TEST SPLIT
# ==========================================================

print("\n===================================")
print("TRAIN-TEST SPLIT")
print("===================================")

# ==========================================================
# TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

print("Training Features:", X_train.shape[1])
print("Testing Features :", X_test.shape[1])


# ==========================================================
# 9. FEATURE SCALING
# ==========================================================

print("\n===================================")
print("FEATURE SCALING")
print("===================================")

scaler = StandardScaler()

# Fit ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

print("Feature Scaling Completed")

print("Scaled Training Shape:",
      X_train_scaled.shape)

print("Scaled Testing Shape :",
      X_test_scaled.shape)


# ==========================================================
# 10. FINAL SUMMARY
# ==========================================================

print("\n===================================")
print("PREPROCESSING COMPLETED")
print("===================================")

print("Final Dataset Shape :", df.shape)

print("X Shape             :", X.shape)

print("y Shape             :", y.shape)

print("X Train Shape       :", X_train_scaled.shape)

print("X Test Shape        :", X_test_scaled.shape)

print("y Train Shape       :", y_train.shape)

print("y Test Shape        :", y_test.shape)

print("Number of Classes   :",
      len(label_encoder.classes_))

# ==========================================================
# SAVE PREPROCESSED DATA
# ==========================================================

PREPROCESSED_PATH = r"D:\AI_Sign_Gesture_System\outputs\preprocessed"

os.makedirs(PREPROCESSED_PATH, exist_ok=True)

joblib.dump(
    X_train,
    os.path.join(PREPROCESSED_PATH, "X_train.pkl")
)

joblib.dump(
    X_test,
    os.path.join(PREPROCESSED_PATH, "X_test.pkl")
)

joblib.dump(
    y_train,
    os.path.join(PREPROCESSED_PATH, "y_train.pkl")
)

joblib.dump(
    y_test,
    os.path.join(PREPROCESSED_PATH, "y_test.pkl")
)

joblib.dump(
    X_train_scaled,
    os.path.join(PREPROCESSED_PATH, "X_train_scaled.pkl")
)

joblib.dump(
    X_test_scaled,
    os.path.join(PREPROCESSED_PATH, "X_test_scaled.pkl")
)

joblib.dump(
    scaler,
    os.path.join(PREPROCESSED_PATH, "scaler.pkl")
)

joblib.dump(
    label_encoder,
    os.path.join(PREPROCESSED_PATH, "label_encoder.pkl")
)

print("\n===================================")
print("PREPROCESSED DATA SAVED")
print("===================================")

print("Saved to:")
print(PREPROCESSED_PATH)