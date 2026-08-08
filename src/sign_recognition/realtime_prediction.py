import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import Counter, deque


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

MODEL_PATH = r"D:\AI_Sign_Gesture_System\outputs\models"
PREPROCESSED_PATH = r"D:\AI_Sign_Gesture_System\outputs\preprocessed"


# ==========================================================
# 2. MODEL FILES
# ==========================================================

MODEL_FILE = os.path.join(
    MODEL_PATH,
    "random_forest.pkl"
)

ENCODER_FILE = os.path.join(
    PREPROCESSED_PATH,
    "label_encoder.pkl"
)


# ==========================================================
# 3. LOAD RANDOM FOREST
# ==========================================================

if not os.path.exists(MODEL_FILE):

    print("\nERROR: Random Forest model not found.")
    print(MODEL_FILE)
    exit()


model = joblib.load(
    MODEL_FILE
)


# ==========================================================
# 4. LOAD LABEL ENCODER
# ==========================================================

if not os.path.exists(ENCODER_FILE):

    print("\nERROR: Label encoder not found.")
    print(ENCODER_FILE)
    exit()


label_encoder = joblib.load(
    ENCODER_FILE
)


# ==========================================================
# 5. STARTUP INFORMATION
# ==========================================================

print("\n==============================================")
print("       GESTUREX SIGN LANGUAGE DETECTION")
print("==============================================")

print("Random Forest       : Loaded")
print("Label Encoder       : Loaded")
print(
    "Number of Classes   :",
    len(label_encoder.classes_)
)

print(
    "Classes             :",
    list(label_encoder.classes_)
)

print("==============================================")


# ==========================================================
# 6. MEDIAPIPE HAND DETECTION
# ==========================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# ==========================================================
# 7. OPEN WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("\nERROR: Could not open webcam.")
    exit()


print("\nWebcam Started Successfully.")

print("\nControls:")
print("SPACE = Add detected sign")
print("B     = Delete last character")
print("C     = Clear sentence")
print("Q     = Quit")


# ==========================================================
# 8. VARIABLES
# ==========================================================

sentence = ""

current_prediction = "No Hand"

current_confidence = 0.0


# Prediction history for stability

prediction_history = deque(
    maxlen=5
)


# ==========================================================
# 9. MAIN LOOP
# ==========================================================

while True:

    # ------------------------------------------------------
    # Capture webcam frame
    # ------------------------------------------------------

    success, frame = cap.read()


    if not success:

        print(
            "\nERROR: Could not read webcam frame."
        )

        break


    # ------------------------------------------------------
    # Mirror webcam
    # ------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # ------------------------------------------------------
    # Convert BGR → RGB
    # ------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ------------------------------------------------------
    # MediaPipe hand detection
    # ------------------------------------------------------

    results = hands.process(
        rgb_frame
    )


    # Default prediction

    current_prediction = "No Hand"

    current_confidence = 0.0


    # ======================================================
    # 10. HAND FOUND
    # ======================================================

    if results.multi_hand_landmarks:

        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )


        # --------------------------------------------------
        # Draw landmarks
        # --------------------------------------------------

        mp_drawing.draw_landmarks(

            frame,

            hand_landmarks,

            mp_hands.HAND_CONNECTIONS
        )


        # ==================================================
        # 11. EXTRACT 21 LANDMARKS
        # ==================================================

        features = []


        for landmark in hand_landmarks.landmark:

            features.append(
                landmark.x
            )

            features.append(
                landmark.y
            )

            features.append(
                landmark.z
            )


        # ==================================================
        # 12. CONVERT TO NUMPY
        # ==================================================

        features = np.array(

            features,

            dtype=np.float32
        )


        # ==================================================
        # 13. VERIFY 63 FEATURES
        # ==================================================

        if len(features) == 63:

            features = features.reshape(
                1,
                63
            )


            # ==============================================
            # 14. RANDOM FOREST PREDICTION
            # ==============================================

            prediction = model.predict(
                features
            )


            # ==============================================
            # 15. DECODE LABEL
            # ==============================================

            predicted_label = (

                label_encoder.inverse_transform(
                    prediction
                )[0]
            )


            # ==============================================
            # 16. CONFIDENCE
            # ==============================================

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (

                    model.predict_proba(
                        features
                    )
                )


                current_confidence = (

                    np.max(
                        probabilities[0]
                    ) * 100
                )


            # ==============================================
            # 17. STABLE PREDICTION
            # ==============================================

            prediction_history.append(
                predicted_label
            )


            common_prediction = (

                Counter(
                    prediction_history
                ).most_common(1)
            )


            if common_prediction:

                current_prediction = (

                    common_prediction[0][0]
                )


        else:

            current_prediction = (
                "Feature Error"
            )


    else:

        prediction_history.clear()


    # ======================================================
    # 18. DISPLAY CURRENT SIGN
    # ======================================================

    cv2.putText(

        frame,

        f"Sign: {current_prediction}",

        (20, 45),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 0),

        2
    )


    # ======================================================
    # 19. DISPLAY CONFIDENCE
    # ======================================================

    if current_confidence > 0:

        cv2.putText(

            frame,

            f"Confidence: {current_confidence:.1f}%",

            (20, 85),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (0, 255, 255),

            2
        )


    # ======================================================
    # 20. DISPLAY RECOGNIZED TEXT
    # ======================================================

    cv2.rectangle(

        frame,

        (15, 110),

        (frame.shape[1] - 15, 175),

        (30, 30, 30),

        -1
    )


    cv2.putText(

        frame,

        "Recognized Text:",

        (25, 140),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        sentence,

        (25, 165),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 0),

        2
    )


    # ======================================================
    # 21. DISPLAY INSTRUCTIONS
    # ======================================================

    cv2.putText(

        frame,

        "SPACE:Add  B:Delete  C:Clear  Q:Quit",

        (20, frame.shape[0] - 20),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        1
    )


    # ======================================================
    # 22. SHOW WINDOW
    # ======================================================

    cv2.imshow(

        "GestureX - Sign Language Detection",

        frame
    )


    # ======================================================
    # 23. KEYBOARD CONTROL
    # ======================================================

    key = cv2.waitKey(1) & 0xFF


    # ------------------------------------------------------
    # SPACE → ADD SIGN
    # ------------------------------------------------------

    if key == ord(" "):

        if (

            current_prediction != "No Hand"

            and current_prediction != "Feature Error"
        ):

            sentence += str(
                current_prediction
            )

            print(
                "Added:",
                current_prediction
            )

            print(
                "Text:",
                sentence
            )


    # ------------------------------------------------------
    # B → DELETE LAST CHARACTER
    # ------------------------------------------------------

    elif key == ord("b"):

        if len(sentence) > 0:

            sentence = sentence[:-1]

            print(
                "Text:",
                sentence
            )


    # ------------------------------------------------------
    # C → CLEAR
    # ------------------------------------------------------

    elif key == ord("c"):

        sentence = ""

        print(
            "Recognized text cleared."
        )


    # ------------------------------------------------------
    # Q → EXIT
    # ------------------------------------------------------

    elif key == ord("q"):

        break


# ==========================================================
# 24. CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()

hands.close()


# ==========================================================
# 25. FINAL OUTPUT
# ==========================================================

print("\n==============================================")
print("       SIGN LANGUAGE DETECTION STOPPED")
print("==============================================")

print(
    "Final Recognized Text:",
    sentence
)

print("==============================================")