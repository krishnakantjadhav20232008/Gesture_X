import os
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)


# ==========================================================
# 1. SETUP DYNAMIC PATHS & DIRECTORIES
# ==========================================================

# Dynamically locate project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate to project root if running inside 'models' or a subfolder
if os.path.basename(SCRIPT_DIR) in ["models", "src", "scripts"]:
    BASE_DIR = os.path.dirname(SCRIPT_DIR)
else:
    BASE_DIR = SCRIPT_DIR

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# FOLDER PATHS (Directories only)
PREPROCESSED_DIR = os.path.join(OUTPUTS_DIR, "preprocessed_gesture")
MODEL_DIR = os.path.join(OUTPUTS_DIR, "model_gesture")

# Create output folder safely if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Project Base Directory : {BASE_DIR}")
print(f"Loading data from      : {PREPROCESSED_DIR}")
print(f"Saving models to       : {MODEL_DIR}")


# ==========================================================
# 2. LOAD PREPROCESSED DATA
# ==========================================================

print("\n===================================")
print("LOADING PREPROCESSED DATA")
print("===================================")

try:
    X_train = joblib.load(os.path.join(PREPROCESSED_DIR, "X_train.pkl"))
    X_test = joblib.load(os.path.join(PREPROCESSED_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(PREPROCESSED_DIR, "y_train.pkl"))
    y_test = joblib.load(os.path.join(PREPROCESSED_DIR, "y_test.pkl"))
    X_train_scaled = joblib.load(os.path.join(PREPROCESSED_DIR, "X_train_scaled.pkl"))
    X_test_scaled = joblib.load(os.path.join(PREPROCESSED_DIR, "X_test_scaled.pkl"))

    print("Training Samples :", X_train.shape[0])
    print("Testing Samples  :", X_test.shape[0])
    print("Features         :", X_train.shape[1])

except FileNotFoundError as e:
    print(f"\n[ERROR] Missing preprocessed data file: {e}")
    print(f"Please ensure preprocessed files exist in: {PREPROCESSED_DIR}")
    exit(1)


# ==========================================================
# 3. CREATE CLASSIFICATION MODELS
# ==========================================================

print("\n===================================")
print("CREATING CLASSIFICATION MODELS")
print("===================================")

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "SVM": SVC(
        kernel="rbf",
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    )
}

print("\nModels Created:")
for model_name in models:
    print("-", model_name)


# ==========================================================
# 4. TRAIN AND EVALUATE MODELS
# ==========================================================

results = {}

for model_name, model in models.items():

    print("\n===================================")
    print("TRAINING:", model_name)
    print("===================================")

    # Decision Tree / Random Forest
    if model_name in ["Decision Tree", "Random Forest"]:
        print("Using original features...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # KNN / SVM / Logistic Regression
    else:
        print("Using scaled features...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    # Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(
        y_test, 
        y_pred, 
        average="weighted", 
        zero_division=0
    )
    confusion = confusion_matrix(y_test, y_pred)

    # Store Results
    results[model_name] = {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": confusion
    }

    # Display Metrics
    print("\nAccuracy :", f"{accuracy:.4f}")
    print("F1 Score :", f"{f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion)

    # Save Individual Model
    model_filename = model_name.lower().replace(" ", "_") + ".pkl"
    model_file = os.path.join(MODEL_DIR, model_filename)

    joblib.dump(model, model_file)
    print("\nModel Saved:", model_file)


# ==========================================================
# 5. MODEL COMPARISON
# ==========================================================

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(f"{'Model':25}{'Accuracy':15}{'F1 Score':15}")
print("-" * 55)

for model_name, result in results.items():
    print(
        f"{model_name:25}"
        f"{result['accuracy']:<15.4f}"
        f"{result['f1_score']:<15.4f}"
    )


# ==========================================================
# 6. FIND BEST MODEL
# ==========================================================

best_model_name = max(
    results,
    key=lambda name: results[name]["f1_score"]
)

best_model = models[best_model_name]
best_accuracy = results[best_model_name]["accuracy"]
best_f1 = results[best_model_name]["f1_score"]


# ==========================================================
# 7. DISPLAY & SAVE BEST MODEL
# ==========================================================

print("\n===================================")
print("BEST MODEL")
print("===================================")

print("Model    :", best_model_name)
print("Accuracy :", f"{best_accuracy:.4f}")
print("F1 Score :", f"{best_f1:.4f}")

# Save Best Model File
best_model_file = os.path.join(MODEL_DIR, "gesturex_best_model.pkl")
joblib.dump(best_model, best_model_file)
print("\nBest Model Saved:", best_model_file)

# Save Model Results File
results_file = os.path.join(MODEL_DIR, "model_results.pkl")
joblib.dump(results, results_file)
print("Model Results Saved:", results_file)


# ==========================================================
# 8. FINAL SUMMARY
# ==========================================================

print("\n===================================")
print("MODEL TRAINING COMPLETED")
print("===================================")

print("\nAll Models Trained & Saved:")
for model_name in models:
    print("✓", model_name)

print("\nTop Performing Model:")
print("✓", best_model_name)

print("\nReady for Real-Time Prediction! 🚀")