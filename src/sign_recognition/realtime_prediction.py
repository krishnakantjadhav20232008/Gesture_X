import os
import cv2
import joblib
import numpy as np
import mediapipe as mp

from collections import Counter, deque


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

MODEL_FILE = (
    r"D:\AI_Sign_Gesture_System"
    r"\outputs\models_signs\random_forest.pkl"
)

ENCODER_FILE = (
    r"D:\AI_Sign_Gesture_System"
    r"\outputs\preprocessed_signs\label_encoder.pkl"
)


# ==========================================================
# 2. CHECK MODEL FILES
# ==========================================================

print("\n==============================================")
print("       GESTUREX SIGN LANGUAGE DETECTION")
print("==============================================")


if not os.path.isfile(MODEL_FILE):

    print("\nERROR: Random Forest model not found.")
    print(MODEL_FILE)
    print("\nPlease check that this file exists.")
    exit()


if not os.path.isfile(ENCODER_FILE):

    print("\nERROR: Label encoder not found.")
    print(ENCODER_FILE)
    print("\nPlease check that this file exists.")
    exit()


# ==========================================================
# 3. LOAD RANDOM FOREST MODEL
# ==========================================================

try:

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Random Forest       : Loaded"
    )

except Exception as error:

    print(
        "\nERROR: Could not load Random Forest."
    )

    print(error)

    exit()


# ==========================================================
# 4. LOAD LABEL ENCODER
# ==========================================================

try:

    label_encoder = joblib.load(
        ENCODER_FILE
    )

    print(
        "Label Encoder       : Loaded"
    )

except Exception as error:

    print(
        "\nERROR: Could not load Label Encoder."
    )

    print(error)

    exit()


# ==========================================================
# 5. MODEL INFORMATION
# ==========================================================

print(
    "Number of Classes   :",
    len(
        label_encoder.classes_
    )
)

print(
    "Classes             :",
    list(
        label_encoder.classes_
    )
)


# ==========================================================
# 6. VERIFY MODEL FEATURES
# ==========================================================

if hasattr(
    model,
    "n_features_in_"
):

    print(
        "Model Features      :",
        model.n_features_in_
    )

    if model.n_features_in_ != 63:

        print(
            "\nWARNING: Model does not expect 63 features."
        )

        print(
            "Expected:",
            model.n_features_in_
        )

        print(
            "Real-time MediaPipe features: 63"
        )


print("==============================================")


# ==========================================================
# 7. MEDIAPIPE HANDS
# ==========================================================

try:

    mp_hands = mp.solutions.hands

    mp_drawing = (
        mp.solutions.drawing_utils
    )

except Exception as error:

    print(
        "\nERROR: MediaPipe could not be loaded."
    )

    print(error)

    exit()


# ==========================================================
# 8. INITIALIZE MEDIAPIPE
# ==========================================================

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# ==========================================================
# 9. OPEN WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print(
        "\nERROR: Could not open webcam."
    )

    hands.close()

    exit()


print(
    "\nWebcam Started Successfully."
)


# ==========================================================
# 10. CONTROLS
# ==========================================================

print("\nControls:")
print(
    "SPACE = Add detected sign"
)
print(
    "B     = Delete last character"
)
print(
    "C     = Clear sentence"
)
print(
    "Q     = Quit"
)

print(
    "=============================================="
)


# ==========================================================
# 11. VARIABLES
# ==========================================================

sentence = ""

current_prediction = (
    "No Hand"
)

current_confidence = 0.0


# ==========================================================
# 12. PREDICTION HISTORY
# ==========================================================

prediction_history = deque(
    maxlen=5
)


# ==========================================================
# 13. MAIN LOOP
# ==========================================================

while True:


    # ======================================================
    # CAPTURE FRAME
    # ======================================================

    success, frame = cap.read()


    if not success:

        print(
            "\nERROR: Could not read webcam frame."
        )

        break


    # ======================================================
    # MIRROR IMAGE
    # ======================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ======================================================
    # BGR → RGB
    # ======================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ======================================================
    # MEDIAPIPE PROCESSING
    # ======================================================

    results = hands.process(
        rgb_frame
    )


    # ======================================================
    # DEFAULT VALUES
    # ======================================================

    current_prediction = (
        "No Hand"
    )

    current_confidence = 0.0


    # ======================================================
    # HAND DETECTED
    # ======================================================

    if results.multi_hand_landmarks:


        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )


        # ==================================================
        # DRAW HAND LANDMARKS
        # ==================================================

        mp_drawing.draw_landmarks(

            frame,

            hand_landmarks,

            mp_hands.HAND_CONNECTIONS
        )


        # ==================================================
        # EXTRACT 21 LANDMARKS
        # ==================================================

        features = []


        for landmark in (
            hand_landmarks.landmark
        ):

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
        # NUMPY ARRAY
        # ==================================================

        features = np.array(

            features,

            dtype=np.float32
        )


        # ==================================================
        # VERIFY 63 FEATURES
        # ==================================================

        if len(features) == 63:


            # ==================================================
            # RESHAPE
            # ==================================================

            landmarks = features.reshape(

                21,

                3
            )


            # ==================================================
            # LANDMARK NORMALIZATION
            #
            # IMPORTANT:
            # This is the SAME preprocessing used
            # in your working Sign Language code.
            # ==================================================


            # --------------------------------------------------
            # WRIST = LANDMARK 0
            # --------------------------------------------------

            wrist = landmarks[0].copy()


            # --------------------------------------------------
            # MOVE WRIST TO ORIGIN
            # --------------------------------------------------

            landmarks = (
                landmarks - wrist
            )


            # --------------------------------------------------
            # CALCULATE HAND SIZE
            # --------------------------------------------------

            distances = np.linalg.norm(

                landmarks,

                axis=1
            )


            scale = np.max(
                distances
            )


            # --------------------------------------------------
            # PREVENT DIVISION BY ZERO
            # --------------------------------------------------

            if scale < 1e-8:

                scale = 1.0


            # --------------------------------------------------
            # NORMALIZE LANDMARKS
            # --------------------------------------------------

            landmarks = (
                landmarks / scale
            )


            # ==================================================
            # FINAL 63-FEATURE INPUT
            # ==================================================

            features = landmarks.reshape(

                1,

                63
            )


            # ==================================================
            # RANDOM FOREST PREDICTION
            # ==================================================

            try:

                prediction = model.predict(
                    features
                )


                # ==================================================
                # LABEL ENCODER
                # ==================================================

                predicted_label = (

                    label_encoder
                    .inverse_transform(
                        prediction
                    )[0]
                )


                # ==================================================
                # CONFIDENCE
                # ==================================================

                current_confidence = 0.0


                if hasattr(

                    model,

                    "predict_proba"

                ):

                    probabilities = (

                        model
                        .predict_proba(
                            features
                        )
                    )


                    current_confidence = (

                        float(
                            np.max(
                                probabilities[0]
                            )
                        )

                        * 100
                    )


                # ==================================================
                # PREDICTION SMOOTHING
                # ==================================================

                prediction_history.append(

                    str(
                        predicted_label
                    )
                )


                common_prediction = (

                    Counter(
                        prediction_history
                    )
                    .most_common(1)
                )


                if common_prediction:

                    current_prediction = (

                        common_prediction[0][0]
                    )

                else:

                    current_prediction = (

                        str(
                            predicted_label
                        )
                    )


            except Exception as error:

                current_prediction = (
                    "Prediction Error"
                )

                current_confidence = 0.0

                print(
                    "\nPrediction Error:",
                    error
                )


        else:

            current_prediction = (
                "Feature Error"
            )

            current_confidence = 0.0


    else:

        # ==================================================
        # NO HAND
        # ==================================================

        prediction_history.clear()

        current_prediction = (
            "No Hand"
        )

        current_confidence = 0.0


    # ======================================================
    # DISPLAY SIGN
    # ======================================================

    cv2.rectangle(

        frame,

        (
            0,
            0
        ),

        (
            frame.shape[1],
            105
        ),

        (
            20,
            20,
            20
        ),

        -1
    )


    cv2.putText(

        frame,

        f"Sign: {current_prediction}",

        (
            20,
            40
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (
            0,
            255,
            0
        ),

        2
    )


    # ======================================================
    # DISPLAY CONFIDENCE
    # ======================================================

    cv2.putText(

        frame,

        f"Confidence: "
        f"{current_confidence:.1f}%",

        (
            20,
            82
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (
            0,
            255,
            255
        ),

        2
    )


    # ======================================================
    # RECOGNIZED TEXT BOX
    # ======================================================

    cv2.rectangle(

        frame,

        (
            15,
            120
        ),

        (
            frame.shape[1] - 15,
            185
        ),

        (
            30,
            30,
            30
        ),

        -1
    )


    cv2.putText(

        frame,

        "Recognized Text:",

        (
            25,
            148
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            255,
            255,
            255
        ),

        2
    )


    cv2.putText(

        frame,

        sentence,

        (
            25,
            175
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (
            0,
            255,
            0
        ),

        2
    )


    # ======================================================
    # INSTRUCTIONS
    # ======================================================

    cv2.putText(

        frame,

        "SPACE:Add  B:Delete  C:Clear  Q:Quit",

        (
            20,
            frame.shape[0] - 20
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (
            255,
            255,
            255
        ),

        1
    )


    # ======================================================
    # SHOW WINDOW
    # ======================================================

    cv2.imshow(

        "GestureX - Sign Language Detection",

        frame
    )


    # ======================================================
    # KEYBOARD INPUT
    # ======================================================

    key = (
        cv2.waitKey(1) & 0xFF
    )


    # ======================================================
    # SPACE → ADD SIGN
    # ======================================================

    if key == ord(" "):

        if (

            current_prediction
            not in [
                "No Hand",
                "Feature Error",
                "Prediction Error"
            ]

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


    # ======================================================
    # B → DELETE LAST CHARACTER
    # ======================================================

    elif key == ord("b"):

        if len(sentence) > 0:

            sentence = (
                sentence[:-1]
            )


            print(
                "Text:",
                sentence
            )


    # ======================================================
    # C → CLEAR
    # ======================================================

    elif key == ord("c"):

        sentence = ""

        print(
            "\nRecognized text cleared."
        )


    # ======================================================
    # Q → QUIT
    # ======================================================

    elif key == ord("q"):

        break


# ==========================================================
# 14. CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()

hands.close()


# ==========================================================
# 15. FINAL OUTPUT
# ==========================================================

print(
    "\n=============================================="
)

print(
    "       SIGN LANGUAGE DETECTION STOPPED"
)

print(
    "=============================================="
)

print(
    "Final Recognized Text:",
    sentence
)

print(
    "=============================================="
)