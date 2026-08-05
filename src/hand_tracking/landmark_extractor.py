import os
import cv2
import csv
import mediapipe as mp

# ==========================
# INITIALIZE MEDIAPIPE
# ==========================
mp_hands = mp.solutions.hands

# ==========================
# DATASET PATH
# ==========================
DATASET_PATH = r"D:\AI_Datasets"

# Output CSV
CSV_FILE = r"D:\AI_Sign_Gesture_System\outputs\gesturex_dataset.csv"

# ==========================
# CREATE CSV HEADER
# ==========================
header = []

# 21 Landmarks × (x, y, z)
for i in range(21):
    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

header.append("label")

# ==========================
# OPEN CSV FILE
# ==========================
with open(CSV_FILE, mode="w", newline="") as file:

    writer = csv.writer(file)
    writer.writerow(header)

    # Initialize MediaPipe
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        # ----------------------------------------
        # Read every dataset
        # Alphabets / Numbers / Gesture
        # ----------------------------------------
        for dataset in os.listdir(DATASET_PATH):

            dataset_folder = os.path.join(DATASET_PATH, dataset)

            if not os.path.isdir(dataset_folder):
                continue

            print(f"\nProcessing Dataset : {dataset}")

            # ----------------------------------------
            # Read every class
            # Example:
            # A, B, C...
            # 0,1,2...
            # Palm, Fist...
            # ----------------------------------------
            for label in os.listdir(dataset_folder):

                label_folder = os.path.join(dataset_folder, label)

                if not os.path.isdir(label_folder):
                    continue

                print(f"Class : {label}")

                # ----------------------------------------
                # Read every image
                # ----------------------------------------
                for image_name in os.listdir(label_folder):

                    image_path = os.path.join(label_folder, image_name)

                    image = cv2.imread(image_path)

                    if image is None:
                        continue

                    # Convert BGR to RGB
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Detect hand
                    results = hands.process(rgb)

                    # Skip image if no hand detected
                    if not results.multi_hand_landmarks:
                        continue

                    # Process first detected hand
                    hand_landmarks = results.multi_hand_landmarks[0]

                    row = []

                    # ----------------------------------------
                    # Extract 21 Landmarks
                    # ----------------------------------------
                    for landmark in hand_landmarks.landmark:

                        row.append(landmark.x)
                        row.append(landmark.y)
                        row.append(landmark.z)

                    # Append Label
                    row.append(label)

                    # Save Row
                    writer.writerow(row)

print("\n===================================")
print("Landmark Extraction Completed!")
print(f"CSV Saved As : {CSV_FILE}")
print("===================================")