import os
import cv2
import csv
import joblib
import numpy as np
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_PATH = r"D:\AI_Sign_Gesture_System"

MODEL_PATH = os.path.join(
    BASE_PATH,
    "outputs",
    "model_gesture"
)

PREPROCESSED_PATH = os.path.join(
    BASE_PATH,
    "outputs",
    "preprocessed_gesture"
)

DATA_PATH = os.path.join(
    BASE_PATH,
    "outputs",
    "gesture_control_dataset.csv"
)

MODEL_FILE = os.path.join(
    MODEL_PATH,
    "gesture_control_rf.pkl"
)

ENCODER_FILE = os.path.join(
    PREPROCESSED_PATH,
    "gesture_encoder.pkl"
)


os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PREPROCESSED_PATH, exist_ok=True)


# ==========================================================
# GESTURE LABELS
# ==========================================================

GESTURES = {
    "1": "open_palm",
    "2": "fist",
    "3": "thumbs_up",
    "4": "thumbs_down",
    "5": "index_up",
    "6": "index_down",
    "7": "index_left",
    "8": "index_right",
    "9": "victory",
    "0": "ok_sign"
}


SAMPLES_PER_GESTURE = 150


# ==========================================================
# MEDIAPIPE
# ==========================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==========================================================
# COLLECT DATA
# ==========================================================

print("\n==============================================")
print("     GESTUREX REAL GESTURE DATA COLLECTOR")
print("==============================================")

print("\nYou will collect REAL hand landmarks.")
print("Hold the gesture clearly in front of camera.")
print("\nControls:")

for key, name in GESTURES.items():
    print(f"{key} = {name}")

print("\nQ = Quit")
print("R = Start selected gesture collection")
print("==============================================")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    exit()


# ==========================================================
# CSV HEADER
# ==========================================================

header = ["label"]

for i in range(21):
    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


# Start fresh dataset
with open(
    DATA_PATH,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)
    writer.writerow(header)


# ==========================================================
# COLLECT EACH GESTURE
# ==========================================================

for key, gesture_name in GESTURES.items():

    print("\n----------------------------------------------")
    print(f"NEXT GESTURE: {gesture_name}")
    print(f"Press {key} in the camera window.")
    print("----------------------------------------------")

    started = False
    samples = 0

    while samples < SAMPLES_PER_GESTURE:

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        # ----------------------------------------------
        # WAIT FOR CORRECT KEY
        # ----------------------------------------------

        if not started:

            cv2.putText(
                frame,
                f"Show: {gesture_name}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Press {key} to start",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------
        # COLLECTION
        # ----------------------------------------------

        if started:

            cv2.putText(
                frame,
                f"Collecting: {gesture_name}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {samples}/{SAMPLES_PER_GESTURE}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------
        # HAND DETECTED
        # ----------------------------------------------

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            if started:

                features = []

                for landmark in hand.landmark:

                    features.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                if len(features) == 63:

                    with open(
                        DATA_PATH,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow(
                            [gesture_name] + features
                        )

                    samples += 1

        cv2.imshow(
            "GestureX - Dataset Collection",
            frame
        )

        key_pressed = cv2.waitKey(1) & 0xFF

        if key_pressed == ord("q"):

            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            exit()

        if not started and key_pressed == ord(key):

            print(
                f"Collecting {gesture_name}..."
            )

            started = True


    print(
        f"Completed: {gesture_name} "
        f"({SAMPLES_PER_GESTURE} samples)"
    )


# ==========================================================
# CLEANUP CAMERA
# ==========================================================

cap.release()
cv2.destroyAllWindows()
hands.close()


# ==========================================================
# LOAD DATASET
# ==========================================================

print("\n==============================================")
print("DATA COLLECTION COMPLETE")
print("==============================================")

data = []

labels = []

with open(
    DATA_PATH,
    "r"
) as file:

    reader = csv.reader(file)

    next(reader)

    for row in reader:

        labels.append(row[0])

        data.append(
            [float(x) for x in row[1:]]
        )


X = np.array(
    data,
    dtype=np.float32
)

y = np.array(
    labels
)


print(
    "Total Samples:",
    len(X)
)

print(
    "Features:",
    X.shape[1]
)


# ==========================================================
# LABEL ENCODING
# ==========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nEncoded Classes:")

for i, name in enumerate(
    label_encoder.classes_
):

    print(
        i,
        "->",
        name
    )


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


# ==========================================================
# RANDOM FOREST
# ==========================================================

print("\nTraining Random Forest...")


model = RandomForestClassifier(

    n_estimators=300,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ==========================================================
# EVALUATION
# ==========================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n==============================================")
print("MODEL EVALUATION")
print("==============================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    )
)


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    MODEL_FILE
)


joblib.dump(
    label_encoder,
    ENCODER_FILE
)


# ==========================================================
# FINAL
# ==========================================================

print("\n==============================================")
print("GESTURE MODEL READY")
print("==============================================")

print(
    "Model saved:"
)

print(
    MODEL_FILE
)

print(
    "\nEncoder saved:"
)

print(
    ENCODER_FILE
)

print(
    "\nDataset saved:"
)

print(
    DATA_PATH
)

print("\n==============================================")