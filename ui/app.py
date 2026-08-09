import os
import time
import ctypes
import threading

import cv2
import joblib
import numpy as np
import mediapipe as mp
import streamlit as st
import av

from collections import deque, Counter

mp_error = None
mp_hands = None
mp_drawing = None

try:
    mp_hands = mp.solutions.hands   
    mp_drawing = mp.solutions.drawing_utils
except Exception as err:
    mp_error = err

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="GestureX",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = r"D:\AI_Sign_Gesture_System"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


# ==========================================================
# GESTURE CONTROL MODEL
# ==========================================================

GESTURE_MODEL_FILE = os.path.join(
    OUTPUTS_DIR,
    "model_gesture",
    "gesture_control_rf.pkl"
)

GESTURE_ENCODER_FILE = os.path.join(
    OUTPUTS_DIR,
    "preprocessed_gesture",
    "gesture_encoder.pkl"
)


# ==========================================================
# SIGN LANGUAGE MODEL
# ==========================================================

SIGN_MODEL_FILE = os.path.join(
    OUTPUTS_DIR,
    "models_signs",
    "random_forest.pkl"
)

SIGN_ENCODER_FILE = os.path.join(
    OUTPUTS_DIR,
    "preprocessed_signs",
    "label_encoder.pkl"
)


# ==========================================================
# 3. CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #888888;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #202020;
        color: white;
        margin-top: 10px;
    }

    .gesture-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #111111;
        color: #00ff00;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
    }

    .confidence-box {
        padding: 12px;
        border-radius: 10px;
        background-color: #202020;
        color: #ffff00;
        font-size: 20px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# 4. TITLE
# ==========================================================

st.markdown(
    '<div class="main-title">🖐️ GestureX</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI Sign Language & Gesture Control System'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# 5. CHECK FILES
# ==========================================================

def check_file(path):

    return os.path.exists(path)


missing_files = []

required_files = [
    GESTURE_MODEL_FILE,
    GESTURE_ENCODER_FILE,
    SIGN_MODEL_FILE,
    SIGN_ENCODER_FILE
]

for file_path in required_files:

    if not check_file(file_path):

        missing_files.append(file_path)


if missing_files:

    st.error("Some required model files are missing.")

    for file_path in missing_files:

        st.code(file_path)

    st.stop()


# ==========================================================
# 6. LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    gesture_model = joblib.load(
        GESTURE_MODEL_FILE
    )

    gesture_encoder = joblib.load(
        GESTURE_ENCODER_FILE
    )

    sign_model = joblib.load(
        SIGN_MODEL_FILE
    )

    sign_encoder = joblib.load(
        SIGN_ENCODER_FILE
    )

    return (
        gesture_model,
        gesture_encoder,
        sign_model,
        sign_encoder
    )


(
    gesture_model,
    gesture_encoder,
    sign_model,
    sign_encoder
) = load_models()


# ==========================================================
# 7. SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ GestureX Control")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Sign Language",
        "Gesture Control"
    ]
)


st.sidebar.markdown("---")

camera_enabled = st.sidebar.checkbox(
    "📷 Camera",
    value=True
)


st.sidebar.markdown("---")


if mode == "Sign Language":

    st.sidebar.success(
        "Sign Language Mode Active"
    )

else:

    st.sidebar.success(
        "Gesture Control Mode Active"
    )


# ==========================================================
# 8. MODEL INFORMATION
# ==========================================================

with st.sidebar.expander(
    "📊 Model Information"
):

    if mode == "Sign Language":

        st.write(
            "Model: Random Forest"
        )

        st.write(
            "Classes:",
            len(
                sign_encoder.classes_
            )
        )

    else:

        st.write(
            "Model: Gesture Control Random Forest"
        )

        st.write(
            "Classes:",
            len(
                gesture_encoder.classes_
            )
        )


# ==========================================================
# 9. GESTURE ACTION MAPPING
# ==========================================================

GESTURE_ACTIONS = {

    # -----------------------------
    # Basic gestures
    # -----------------------------

    "palm":
        "General / Palm",

    "fist":
        "Play / Pause",

    "thumbs-up":
        "Volume Up",

    "thumbs-down":
        "Volume Down",

    # -----------------------------
    # Index gestures
    # -----------------------------

    "index-right":
        "Move Right",

    "index-left":
        "Move Left",

    # -----------------------------
    # Additional gestures
    # -----------------------------

    "volume-up":
        "Volume Up",

    "volume-down":
        "Volume Down",

    "no-gesture":
        "No Action",

    "okay":
        "Enter / OK"
}


# ==========================================================
# 10. WINDOWS MEDIA CONTROL
# ==========================================================

def windows_media_key(vk_code):

    """
    Sends a Windows media key.
    """

    try:

        ctypes.windll.user32.keybd_event(
            vk_code,
            0,
            0,
            0
        )

        ctypes.windll.user32.keybd_event(
            vk_code,
            0,
            2,
            0
        )

        return True

    except Exception:

        return False


def volume_up():

    # Windows VK_VOLUME_UP
    return windows_media_key(0xAF)


def volume_down():

    # Windows VK_VOLUME_DOWN
    return windows_media_key(0xAE)


def play_pause():

    # Windows VK_MEDIA_PLAY_PAUSE
    return windows_media_key(0xB3)


def press_enter():

    # Windows Enter
    return windows_media_key(0x0D)


# ==========================================================
# 11. NORMALIZE LABEL
# ==========================================================

def normalize_label(label):

    label = str(label)

    label = label.lower()

    label = label.strip()

    label = label.replace(
        "_",
        "-"
    )

    label = label.replace(
        " ",
        "-"
    )

    return label


# ==========================================================
# 12. EXECUTE GESTURE ACTION
# ==========================================================

def execute_gesture_action(
    label,
    processor
):

    normalized = normalize_label(
        label
    )

    current_time = time.time()

    # --------------------------------
    # Cooldown
    # --------------------------------

    if (
        current_time -
        processor.last_action_time
        <
        processor.action_cooldown
    ):

        return

    # --------------------------------
    # Volume Up
    # --------------------------------

    if normalized in [
        "thumbs-up",
        "volume-up"
    ]:

        volume_up()

        processor.last_action = (
            "Volume Up"
        )

        processor.last_action_time = (
            current_time
        )

    # --------------------------------
    # Volume Down
    # --------------------------------

    elif normalized in [
        "thumbs-down",
        "volume-down"
    ]:

        volume_down()

        processor.last_action = (
            "Volume Down"
        )

        processor.last_action_time = (
            current_time
        )

    # --------------------------------
    # Index Right
    # --------------------------------

    elif normalized in [
        "index-right",
        "indexright",
        "right"
    ]:

        processor.last_action = (
            "Move Right"
        )

        processor.last_action_time = (
            current_time
        )

    # --------------------------------
    # Index Left
    # --------------------------------

    elif normalized in [
        "index-left",
        "indexleft",
        "left"
    ]:

        processor.last_action = (
            "Move Left"
        )

        processor.last_action_time = (
            current_time
        )

    # --------------------------------
    # Fist
    # --------------------------------

    elif normalized == "fist":

        play_pause()

        processor.last_action = (
            "Play / Pause"
        )

        processor.last_action_time = (
            current_time
        )

    # --------------------------------
    # OK
    # --------------------------------

    elif normalized in [
        "okay",
        "ok"
    ]:

        press_enter()

        processor.last_action = (
            "OK / Enter"
        )

        processor.last_action_time = (
            current_time
        )

    # --------------------------------
    # Palm
    # --------------------------------

    elif normalized == "palm":

        processor.last_action = (
            "Palm Detected"
        )

        processor.last_action_time = (
            current_time
        )

    # --------------------------------
    # No gesture
    # --------------------------------

    elif normalized in [
        "no-gesture",
        "nogesture",
        "none"
    ]:

        processor.last_action = (
            "No Action"
        )


# ==========================================================
# 13. MEDIAPIPE
# ==========================================================

if mp_hands is None or mp_drawing is None:
    st.error(
        "MediaPipe `solutions` API is not available in the installed package.\n"
        "Please install a compatible MediaPipe version, for example `mediapipe==0.10.0`, and restart the app."
    )
    if mp_error is not None:
        st.code(str(mp_error))
    st.stop()


# ==========================================================
# 14. VIDEO PROCESSOR
# ==========================================================

class GestureXProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.mode = mode

        self.lock = threading.Lock()

        self.prediction_history = deque(
            maxlen=7
        )

        self.current_prediction = (
            "No Hand"
        )

        self.current_confidence = 0.0

        self.last_action = (
            "Waiting..."
        )

        self.last_action_time = 0

        self.action_cooldown = 1.0

        self.sentence = ""

        self.hands = mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=1,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5
        )


    # ======================================================
    # EXTRACT FEATURES
    # ======================================================

    def extract_features(
        self,
        hand_landmarks
    ):

        features = []

        for landmark in (
            hand_landmarks.landmark
        ):

            features.extend(
                [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]
            )

        features = np.array(
            features,
            dtype=np.float32
        )

        return features


    # ======================================================
    # PREDICT
    # ======================================================

    def predict(
        self,
        features
    ):

        if self.mode == "Gesture Control":

            model = gesture_model

            encoder = gesture_encoder

        else:

            model = sign_model

            encoder = sign_encoder


        features = features.reshape(
            1,
            63
        )


        prediction = model.predict(
            features
        )


        label = encoder.inverse_transform(
            prediction
        )[0]


        confidence = 0.0


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
                )
                *
                100
            )


        return (
            str(label),
            confidence
        )


    # ======================================================
    # VIDEO FRAME
    # ======================================================

    def recv(
        self,
        frame
    ):

        image = frame.to_ndarray(
            format="bgr24"
        )


        # -----------------------------------------------
        # Mirror image
        # -----------------------------------------------

        image = cv2.flip(
            image,
            1
        )


        # -----------------------------------------------
        # RGB
        # -----------------------------------------------

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # -----------------------------------------------
        # MediaPipe
        # -----------------------------------------------

        results = self.hands.process(
            rgb
        )


        prediction = "No Hand"

        confidence = 0.0


        # =================================================
        # HAND DETECTED
        # =================================================

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )


            # ---------------------------------------------
            # Draw landmarks
            # ---------------------------------------------

            mp_drawing.draw_landmarks(

                image,

                hand_landmarks,

                mp_hands.HAND_CONNECTIONS
            )


            # ---------------------------------------------
            # Extract 63 features
            # ---------------------------------------------

            features = self.extract_features(
                hand_landmarks
            )


            if len(features) == 63:

                try:

                    prediction, confidence = (
                        self.predict(
                            features
                        )
                    )

                except Exception:

                    prediction = (
                        "Prediction Error"
                    )

                    confidence = 0.0


                # -----------------------------------------
                # Smoothing
                # -----------------------------------------

                self.prediction_history.append(
                    prediction
                )


                common = (
                    Counter(
                        self.prediction_history
                    )
                    .most_common(1)
                )


                if common:

                    prediction = (
                        common[0][0]
                    )


                # -----------------------------------------
                # Gesture control
                # -----------------------------------------

                if self.mode == "Gesture Control":

                    execute_gesture_action(
                        prediction,
                        self
                    )


        else:

            self.prediction_history.clear()

            prediction = "No Hand"

            confidence = 0.0


        # =================================================
        # STORE RESULT
        # =================================================

        with self.lock:

            self.current_prediction = (
                prediction
            )

            self.current_confidence = (
                confidence
            )


        # =================================================
        # DISPLAY TOP LABEL
        # =================================================

        cv2.rectangle(

            image,

            (
                0,
                0
            ),

            (
                image.shape[1],
                115
            ),

            (
                20,
                20,
                20
            ),

            -1
        )


        # -----------------------------------------------
        # Mode
        # -----------------------------------------------

        cv2.putText(

            image,

            self.mode,

            (
                20,
                30
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (
                255,
                255,
                255
            ),

            2
        )


        # -----------------------------------------------
        # Prediction
        # -----------------------------------------------

        cv2.putText(

            image,

            f"Sign: {prediction}",

            (
                20,
                65
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


        # -----------------------------------------------
        # Confidence
        # -----------------------------------------------

        cv2.putText(

            image,

            f"Confidence: {confidence:.1f}%",

            (
                20,
                100
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (
                0,
                255,
                255
            ),

            2
        )


        # =================================================
        # GESTURE ACTION
        # =================================================

        if self.mode == "Gesture Control":

            cv2.rectangle(

                image,

                (
                    15,
                    125
                ),

                (
                    image.shape[1] - 15,
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

                image,

                "Gesture Action:",

                (
                    25,
                    150
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (
                    255,
                    255,
                    255
                ),

                2
            )


            cv2.putText(

                image,

                self.last_action,

                (
                    25,
                    178
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (
                    0,
                    255,
                    0
                ),

                2
            )


        return av.VideoFrame.from_ndarray(

            image,

            format="bgr24"
        )


# ==========================================================
# 15. RTC CONFIGURATION
# ==========================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


# ==========================================================
# 16. MAIN UI
# ==========================================================

if not camera_enabled:

    st.info(
        "📷 Camera is OFF. Turn on Camera from the sidebar."
    )

else:

    # ------------------------------------------------------
    # Mode heading
    # ------------------------------------------------------

    if mode == "Sign Language":

        st.subheader(
            "🤟 Sign Language Recognition"
        )

        st.write(
            "Show a trained sign to the camera."
        )

    else:

        st.subheader(
            "🖱️ Gesture Control"
        )

        st.write(
            "Use trained hand gestures to control Windows."
        )


    # ------------------------------------------------------
    # Camera
    # ------------------------------------------------------

    ctx = webrtc_streamer(

        key="gesturex-camera",

        mode=WebRtcMode.SENDRECV,

        rtc_configuration=RTC_CONFIGURATION,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        video_processor_factory=(
            GestureXProcessor
        ),

        async_processing=True
    )


# ==========================================================
# 17. GESTURE CONTROL TABLE
# ==========================================================

if mode == "Gesture Control":

    st.markdown("---")

    st.subheader(
        "🎮 Gesture Controls"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            """
            **Palm** → General gesture

            **Fist** → Play / Pause

            **Thumbs Up** → Volume Up

            **Thumbs Down** → Volume Down

            **Index Right** → Move Right
            """
        )


    with col2:

        st.markdown(
            """
            **Index Left** → Move Left

            **Volume Up** → Volume Up

            **Volume Down** → Volume Down

            **OK** → Enter / OK

            **No Gesture** → No Action
            """
        )


# ==========================================================
# 18. SIGN LANGUAGE INFORMATION
# ==========================================================

else:

    st.markdown("---")

    st.subheader(
        "🤟 Sign Language"
    )

    st.info(
        "The Sign Language Random Forest model "
        "is loaded independently from the Gesture "
        "Control model."
    )


# ==========================================================
# 19. MODEL PATH INFORMATION
# ==========================================================

with st.expander(
    "📁 Loaded Model Paths"
):

    st.write(
        "### Gesture Control"
    )

    st.code(
        GESTURE_MODEL_FILE
    )

    st.code(
        GESTURE_ENCODER_FILE
    )


    st.write(
        "### Sign Language"
    )

    st.code(
        SIGN_MODEL_FILE
    )

    st.code(
        SIGN_ENCODER_FILE
    )


# ==========================================================
# 20. FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#888;">
    GestureX — AI Sign Language & Gesture Control System
    </div>
    """,
    unsafe_allow_html=True
)