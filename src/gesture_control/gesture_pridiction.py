import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
import pyautogui
import ctypes
import time

from collections import Counter, deque


# ==========================================================
# 1. PROJECT PATHS
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


MODEL_FILE = os.path.join(
    MODEL_PATH,
    "gesture_control_rf.pkl"
)

ENCODER_FILE = os.path.join(
    PREPROCESSED_PATH,
    "gesture_encoder.pkl"
)


# ==========================================================
# 2. CHECK MODEL FILES
# ==========================================================

if not os.path.exists(MODEL_FILE):
    print("\nERROR: Gesture model not found:")
    print(MODEL_FILE)
    exit()


if not os.path.exists(ENCODER_FILE):
    print("\nERROR: Gesture encoder not found:")
    print(ENCODER_FILE)
    exit()


# ==========================================================
# 3. LOAD MODEL
# ==========================================================

model = joblib.load(MODEL_FILE)

label_encoder = joblib.load(ENCODER_FILE)


# ==========================================================
# 4. PYAutoGUI SETTINGS
# ==========================================================

pyautogui.PAUSE = 0.05


# ==========================================================
# 5. WINDOWS VOLUME KEYS
# ==========================================================

VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_VOLUME_MUTE = 0xAD


def press_windows_key(key_code):

    ctypes.windll.user32.keybd_event(
        key_code,
        0,
        0,
        0
    )

    ctypes.windll.user32.keybd_event(
        key_code,
        0,
        2,
        0
    )


def volume_down():
    press_windows_key(VK_VOLUME_DOWN)


def volume_up():
    press_windows_key(VK_VOLUME_UP)


def volume_mute():
    press_windows_key(VK_VOLUME_MUTE)


# ==========================================================
# 6. STARTUP INFORMATION
# ==========================================================

print("\n==============================================")
print("       GESTUREX WINDOWS CONTROL")
print("==============================================")

print("Random Forest : Loaded")
print("Encoder       : Loaded")

print(
    "Classes:",
    list(label_encoder.classes_)
)

print("\nGESTURES")

print("1. Open Palm    -> Play / Pause")
print("2. Fist         -> Left Click")
print("3. Thumbs Up    -> Enter")
print("4. Thumbs Down  -> Escape")
print("5. Index Up     -> Scroll Up")
print("6. Index Down   -> Scroll Down")
print("7. Index Left   -> Volume Down")
print("8. Index Right  -> Volume Up")
print("9. Victory      -> Windows + Tab")
print("10. OK Sign     -> Right Click")

print("\nQ = Quit")

print("==============================================\n")


# ==========================================================
# 7. MEDIAPIPE
# ==========================================================

mp_hands = mp.solutions.hands

mp_drawing = (
    mp.solutions.drawing_utils
)


hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=1,

    min_detection_confidence=0.6,

    min_tracking_confidence=0.6
)


# ==========================================================
# 8. WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")

    hands.close()

    exit()


# ==========================================================
# 9. PREDICTION SETTINGS
# ==========================================================

prediction_history = deque(
    maxlen=7
)

CONFIDENCE_LIMIT = 70

ACTION_COOLDOWN = 1.0

last_action = None

last_action_time = 0


# ==========================================================
# 10. ACTION EXECUTOR
# ==========================================================

def execute_action(gesture):

    global last_action
    global last_action_time

    now = time.time()

    # Prevent repeated actions
    # while holding the same gesture

    if (
        gesture == last_action
        and
        now - last_action_time < ACTION_COOLDOWN
    ):
        return ""


    # ------------------------------------------------------
    # OPEN PALM
    # Play / Pause
    # ------------------------------------------------------

    if gesture == "open_palm":

        pyautogui.press(
            "playpause"
        )

        last_action = gesture
        last_action_time = now

        print("ACTION: Play / Pause")

        return "Play / Pause"


    # ------------------------------------------------------
    # FIST
    # Left Click
    # ------------------------------------------------------

    elif gesture == "fist":

        pyautogui.click()

        last_action = gesture
        last_action_time = now

        print("ACTION: Left Click")

        return "Left Click"


    # ------------------------------------------------------
    # THUMBS UP
    # Enter
    # ------------------------------------------------------

    elif gesture == "thumbs_up":

        pyautogui.press(
            "enter"
        )

        last_action = gesture
        last_action_time = now

        print("ACTION: Enter")

        return "Enter"


    # ------------------------------------------------------
    # THUMBS DOWN
    # Escape
    # ------------------------------------------------------

    elif gesture == "thumbs_down":

        pyautogui.press(
            "esc"
        )

        last_action = gesture
        last_action_time = now

        print("ACTION: Escape")

        return "Escape"


    # ------------------------------------------------------
    # INDEX UP
    # Scroll Up
    # ------------------------------------------------------

    elif gesture == "index_up":

        pyautogui.scroll(
            5
        )

        last_action = gesture
        last_action_time = now

        print("ACTION: Scroll Up")

        return "Scroll Up"


    # ------------------------------------------------------
    # INDEX DOWN
    # Scroll Down
    # ------------------------------------------------------

    elif gesture == "index_down":

        pyautogui.scroll(
            -5
        )

        last_action = gesture
        last_action_time = now

        print("ACTION: Scroll Down")

        return "Scroll Down"


    # ------------------------------------------------------
    # INDEX LEFT
    # Volume Down
    # ------------------------------------------------------

    elif gesture == "index_left":

        volume_down()

        last_action = gesture
        last_action_time = now

        print("ACTION: Volume Down")

        return "Volume Down"


    # ------------------------------------------------------
    # INDEX RIGHT
    # Volume Up
    # ------------------------------------------------------

    elif gesture == "index_right":

        volume_up()

        last_action = gesture
        last_action_time = now

        print("ACTION: Volume Up")

        return "Volume Up"


    # ------------------------------------------------------
    # VICTORY
    # Windows + Tab
    # ------------------------------------------------------

    elif gesture == "victory":

        pyautogui.hotkey(
            "win",
            "tab"
        )

        last_action = gesture
        last_action_time = now

        print("ACTION: Windows + Tab")

        return "Windows + Tab"


    # ------------------------------------------------------
    # OK SIGN
    # Right Click
    # ------------------------------------------------------

    elif gesture == "ok_sign":

        pyautogui.rightClick()

        last_action = gesture
        last_action_time = now

        print("ACTION: Right Click")

        return "Right Click"


    return ""


# ==========================================================
# 11. MAIN LOOP
# ==========================================================

while True:

    success, frame = cap.read()


    if not success:

        print(
            "ERROR: Could not read webcam."
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
    # BGR → RGB
    # ------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # ------------------------------------------------------
    # MediaPipe
    # ------------------------------------------------------

    results = hands.process(
        rgb_frame
    )


    predicted_label = "No Hand"

    confidence = 0.0

    action_text = "Waiting..."


    # ======================================================
    # HAND DETECTED
    # ======================================================

    if results.multi_hand_landmarks:

        hand = (
            results.multi_hand_landmarks[0]
        )


        # --------------------------------------------------
        # Draw landmarks
        # --------------------------------------------------

        mp_drawing.draw_landmarks(

            frame,

            hand,

            mp_hands.HAND_CONNECTIONS
        )


        # --------------------------------------------------
        # Extract 63 landmarks
        # --------------------------------------------------

        features = []


        for landmark in hand.landmark:

            features.extend([

                landmark.x,

                landmark.y,

                landmark.z

            ])


        features = np.array(

            features,

            dtype=np.float32
        )


        # --------------------------------------------------
        # Verify features
        # --------------------------------------------------

        if len(features) == 63:

            features = features.reshape(
                1,
                63
            )


            # ----------------------------------------------
            # Random Forest
            # ----------------------------------------------

            prediction = model.predict(
                features
            )


            current_label = (

                label_encoder.inverse_transform(
                    prediction
                )[0]
            )


            # ----------------------------------------------
            # Probability
            # ----------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(
                        features
                    )
                )

                confidence = (

                    float(
                        np.max(
                            probabilities[0]
                        )
                    ) * 100
                )


            # ----------------------------------------------
            # Prediction smoothing
            # ----------------------------------------------

            prediction_history.append(
                current_label
            )


            common = Counter(
                prediction_history
            ).most_common(1)


            if common:

                predicted_label = (
                    common[0][0]
                )


        else:

            predicted_label = "Feature Error"


    else:

        prediction_history.clear()

        last_action = None


    # ======================================================
    # EXECUTE ACTION
    # ======================================================

    if (
        predicted_label != "No Hand"
        and
        predicted_label != "Feature Error"
        and
        confidence >= CONFIDENCE_LIMIT
    ):

        action_result = execute_action(
            predicted_label
        )

        if action_result:

            action_text = action_result

    else:

        if predicted_label != "No Hand":

            action_text = "Low Confidence"


    # ======================================================
    # DISPLAY GESTURE
    # ======================================================

    cv2.putText(

        frame,

        f"Gesture: {predicted_label}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 0),

        2
    )


    # ======================================================
    # DISPLAY CONFIDENCE
    # ======================================================

    cv2.putText(

        frame,

        f"Confidence: {confidence:.1f}%",

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 255),

        2
    )


    # ======================================================
    # ACTION DISPLAY
    # ======================================================

    cv2.rectangle(

        frame,

        (15, 95),

        (
            frame.shape[1] - 15,
            160
        ),

        (30, 30, 30),

        -1
    )


    cv2.putText(

        frame,

        "System Action:",

        (25, 125),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        action_text,

        (25, 150),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (0, 255, 0),

        2
    )


    # ======================================================
    # INSTRUCTIONS
    # ======================================================

    cv2.putText(

        frame,

        "Q = Quit",

        (20, frame.shape[0] - 20),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        1
    )


    # ======================================================
    # SHOW
    # ======================================================

    cv2.imshow(

        "GestureX - Windows Gesture Control",

        frame
    )


    # ======================================================
    # KEYBOARD
    # ======================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ==========================================================
# 12. CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()

hands.close()


print("\n==============================================")
print("       GESTURE CONTROL STOPPED")
print("==============================================")