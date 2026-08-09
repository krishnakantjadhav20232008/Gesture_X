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
    initial_sidebar_state="expanded",
)


# ==========================================================
# 2. PROJECT PATHS
# ==========================================================

BASE_DIR = r"D:\AI_Sign_Gesture_System"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


# ==========================================================
# 3. GESTURE CONTROL MODEL PATHS
# ==========================================================

GESTURE_MODEL_FILE = os.path.join(
    OUTPUTS_DIR,
    "model_gesture",
    "gesture_control_rf.pkl",
)

GESTURE_ENCODER_FILE = os.path.join(
    OUTPUTS_DIR,
    "preprocessed_gesture",
    "gesture_encoder.pkl",
)

GESTURE_SCALER_FILE = os.path.join(
    OUTPUTS_DIR,
    "preprocessed_gesture",
    "scaler.pkl",
)


# ==========================================================
# 4. SIGN LANGUAGE MODEL PATHS
# ==========================================================

SIGN_MODEL_FILE = os.path.join(
    OUTPUTS_DIR,
    "models_signs",
    "random_forest.pkl",
)

SIGN_ENCODER_FILE = os.path.join(
    OUTPUTS_DIR,
    "preprocessed_signs",
    "label_encoder.pkl",
)


# IMPORTANT:
# We DO NOT use sign_scaler.pkl here.
#
# Your verified working Sign Language prediction program
# uses:
#
# 21 landmarks
#       ↓
# wrist moved to origin
#       ↓
# hand-size normalization
#       ↓
# 63 features
#       ↓
# Random Forest
#
# Therefore the exact same preprocessing is used below.


# ==========================================================
# 5. PAGE CSS
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

    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 6. TITLE
# ==========================================================

st.markdown(
    '<div class="main-title">🖐️ GestureX</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    "AI Sign Language & Gesture Control System"
    "</div>",
    unsafe_allow_html=True,
)


# ==========================================================
# 7. CHECK REQUIRED FILES
# ==========================================================

required_files = [
    GESTURE_MODEL_FILE,
    GESTURE_ENCODER_FILE,
    SIGN_MODEL_FILE,
    SIGN_ENCODER_FILE,
]

missing_files = []

for file_path in required_files:

    if not os.path.isfile(file_path):

        missing_files.append(file_path)


if missing_files:

    st.error(
        "❌ Required model files are missing."
    )

    st.write(
        "Please check these paths:"
    )

    for file_path in missing_files:

        st.code(file_path)

    st.stop()


# ==========================================================
# 8. LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    # ------------------------------------------------------
    # Gesture Control
    # ------------------------------------------------------

    gesture_model = joblib.load(
        GESTURE_MODEL_FILE
    )

    gesture_encoder = joblib.load(
        GESTURE_ENCODER_FILE
    )

    gesture_scaler = None

    if os.path.isfile(
        GESTURE_SCALER_FILE
    ):

        gesture_scaler = joblib.load(
            GESTURE_SCALER_FILE
        )


    # ------------------------------------------------------
    # Sign Language
    # ------------------------------------------------------

    sign_model = joblib.load(
        SIGN_MODEL_FILE
    )

    sign_encoder = joblib.load(
        SIGN_ENCODER_FILE
    )


    return (
        gesture_model,
        gesture_encoder,
        gesture_scaler,
        sign_model,
        sign_encoder,
    )


try:

    (
        gesture_model,
        gesture_encoder,
        gesture_scaler,
        sign_model,
        sign_encoder,
    ) = load_models()

except Exception as error:

    st.error(
        "❌ Error while loading model files."
    )

    st.code(
        str(error)
    )

    st.stop()


# ==========================================================
# 9. SIDEBAR
# ==========================================================

st.sidebar.title(
    "⚙️ GestureX Control"
)


mode = st.sidebar.radio(
    "Select Mode",
    [
        "Sign Language",
        "Gesture Control",
    ],
)


camera_enabled = st.sidebar.checkbox(
    "📷 Camera",
    value=True,
)


st.sidebar.markdown("---")


if mode == "Sign Language":

    st.sidebar.success(
        "🤟 Sign Language Mode Active"
    )

else:

    st.sidebar.success(
        "🖱️ Gesture Control Mode Active"
    )


# ==========================================================
# 10. MODEL INFORMATION
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
            ),
        )

        st.write(
            "Preprocessing:"
        )

        st.write(
            "Wrist normalization + "
            "hand-size normalization"
        )

    else:

        st.write(
            "Model: Gesture Control Random Forest"
        )

        st.write(
            "Classes:",
            len(
                gesture_encoder.classes_
            ),
        )

        st.write(
            "Scaler:",
            "Loaded"
            if gesture_scaler is not None
            else "Not Found",
        )


# ==========================================================
# 11. LABEL NORMALIZATION
# ==========================================================

def normalize_label(label):

    label = str(label)

    label = label.strip()

    label = label.lower()

    label = label.replace(
        "_",
        "-",
    )

    label = label.replace(
        " ",
        "-",
    )

    return label


# ==========================================================
# 12. WINDOWS KEY CONTROL
# ==========================================================

def press_key(vk_code):

    try:

        ctypes.windll.user32.keybd_event(
            vk_code,
            0,
            0,
            0,
        )

        ctypes.windll.user32.keybd_event(
            vk_code,
            0,
            2,
            0,
        )

        return True

    except Exception:

        return False


# ==========================================================
# 13. WINDOWS VIRTUAL KEY CODES
# ==========================================================

VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28

VK_ENTER = 0x0D

VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF

VK_MEDIA_PLAY_PAUSE = 0xB3


# ==========================================================
# 14. WINDOWS ACTION FUNCTIONS
# ==========================================================

def volume_up():

    return press_key(
        VK_VOLUME_UP
    )


def volume_down():

    return press_key(
        VK_VOLUME_DOWN
    )


def move_left():

    return press_key(
        VK_LEFT
    )


def move_right():

    return press_key(
        VK_RIGHT
    )


def move_up():

    return press_key(
        VK_UP
    )


def move_down():

    return press_key(
        VK_DOWN
    )


def play_pause():

    return press_key(
        VK_MEDIA_PLAY_PAUSE
    )


def press_enter():

    return press_key(
        VK_ENTER
    )


# ==========================================================
# 15. GESTURE ACTION EXECUTION
# ==========================================================

def execute_gesture_action(
    label,
    processor,
):

    normalized = normalize_label(
        label
    )

    current_time = time.time()


    # ------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------

    if (
        current_time
        - processor.last_action_time
        <
        processor.action_cooldown
    ):

        return


    success = False

    action = "No Action"


    # ======================================================
    # VOLUME UP
    # ======================================================

    if normalized in [

        "thumbs-up",
        "thumbsup",
        "thumb-up",
        "volume-up",
        "volumeup",

    ]:

        success = volume_up()

        action = "🔊 Volume Up"


    # ======================================================
    # VOLUME DOWN
    # ======================================================

    elif normalized in [

        "thumbs-down",
        "thumbsdown",
        "thumb-down",
        "volume-down",
        "volumedown",

    ]:

        success = volume_down()

        action = "🔉 Volume Down"


    # ======================================================
    # RIGHT
    # ======================================================

    elif normalized in [

        "index-right",
        "indexright",
        "right",
        "move-right",
        "moveright",

    ]:

        success = move_right()

        action = "➡️ Move Right"


    # ======================================================
    # LEFT
    # ======================================================

    elif normalized in [

        "index-left",
        "indexleft",
        "left",
        "move-left",
        "moveleft",

    ]:

        success = move_left()

        action = "⬅️ Move Left"


    # ======================================================
    # UP
    # ======================================================

    elif normalized in [

        "up",
        "move-up",
        "moveup",

    ]:

        success = move_up()

        action = "⬆️ Move Up"


    # ======================================================
    # DOWN
    # ======================================================

    elif normalized in [

        "down",
        "move-down",
        "movedown",

    ]:

        success = move_down()

        action = "⬇️ Move Down"


    # ======================================================
    # FIST
    # ======================================================

    elif normalized in [

        "fist",
        "closed-fist",
        "closedfist",

    ]:

        success = play_pause()

        action = "⏯️ Play / Pause"


    # ======================================================
    # OKAY
    # ======================================================

    elif normalized in [

        "okay",
        "ok",
        "ok-sign",
        "oksign",

    ]:

        success = press_enter()

        action = "⏎ Enter / OK"


    # ======================================================
    # PALM
    # ======================================================

    elif normalized in [

        "palm",
        "open-palm",
        "openpalm",

    ]:

        success = True

        action = "✋ Palm Detected"


    # ======================================================
    # NO GESTURE
    # ======================================================

    elif normalized in [

        "no-gesture",
        "nogesture",
        "none",
        "no-action",

    ]:

        return


    # ======================================================
    # SAVE ACTION
    # ======================================================

    if success:

        processor.last_action = action

        processor.last_action_time = (
            current_time
        )


# ==========================================================
# 16. MEDIAPIPE
# ==========================================================

try:

    mp_hands = mp.solutions.hands

    mp_drawing = (
        mp.solutions.drawing_utils
    )

except Exception as error:

    st.error(
        "❌ MediaPipe could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


# ==========================================================
# 17. VIDEO PROCESSOR
# ==========================================================

class GestureXProcessor(
    VideoProcessorBase
):


    # ======================================================
    # INITIALIZATION
    # ======================================================

    def __init__(self):

        self.mode = mode

        self.lock = threading.Lock()


        # --------------------------------------------------
        # MediaPipe
        # --------------------------------------------------

        self.hands = mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=1,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5,

        )


        # --------------------------------------------------
        # Prediction smoothing
        # --------------------------------------------------

        self.prediction_history = deque(
            maxlen=5
        )


        # --------------------------------------------------
        # Results
        # --------------------------------------------------

        self.current_prediction = (
            "No Hand"
        )

        self.current_confidence = 0.0


        # --------------------------------------------------
        # Gesture action
        # --------------------------------------------------

        self.last_action = (
            "Waiting..."
        )

        self.last_action_time = 0

        self.action_cooldown = 1.2


    # ======================================================
    # EXTRACT RAW 63 FEATURES
    # ======================================================

    def extract_features(
        self,
        hand_landmarks,
    ):

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


        return np.asarray(
            features,
            dtype=np.float32,
        )


    # ======================================================
    # SIGN LANGUAGE PREPROCESSING
    # ======================================================

    def preprocess_sign_language(
        self,
        features,
    ):

        # --------------------------------------------------
        # Verify 63 features
        # --------------------------------------------------

        if len(features) != 63:

            raise ValueError(
                "Sign Language requires "
                "exactly 63 features."
            )


        # --------------------------------------------------
        # 63 → 21 × 3
        # --------------------------------------------------

        landmarks = features.reshape(
            21,
            3,
        )


        # --------------------------------------------------
        # LANDMARK 0 = WRIST
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
            axis=1,
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


        # --------------------------------------------------
        # RETURN 63 FEATURES
        # --------------------------------------------------

        return landmarks.reshape(
            1,
            63,
        )


    # ======================================================
    # GESTURE CONTROL PREPROCESSING
    # ======================================================

    def preprocess_gesture(
        self,
        features,
    ):

        if len(features) != 63:

            raise ValueError(
                "Gesture Control requires "
                "exactly 63 features."
            )


        features = features.reshape(
            1,
            63,
        )


        # --------------------------------------------------
        # Gesture model uses its scaler
        # --------------------------------------------------

        if gesture_scaler is not None:

            features = (
                gesture_scaler.transform(
                    features
                )
            )


        return features


    # ======================================================
    # PREDICTION
    # ======================================================

    def predict(
        self,
        features,
    ):


        # ==================================================
        # SIGN LANGUAGE
        # ==================================================

        if self.mode == "Sign Language":

            model = sign_model

            encoder = sign_encoder


            # IMPORTANT:
            # DO NOT use sign_scaler.
            #
            # Use the exact preprocessing from
            # the verified working prediction file.

            model_features = (
                self.preprocess_sign_language(
                    features
                )
            )


        # ==================================================
        # GESTURE CONTROL
        # ==================================================

        else:

            model = gesture_model

            encoder = gesture_encoder


            model_features = (
                self.preprocess_gesture(
                    features
                )
            )


        # ==================================================
        # RANDOM FOREST
        # ==================================================

        prediction = model.predict(
            model_features
        )


        # ==================================================
        # LABEL ENCODER
        # ==================================================

        label = (
            encoder.inverse_transform(
                prediction
            )[0]
        )


        # ==================================================
        # CONFIDENCE
        # ==================================================

        confidence = 0.0


        if hasattr(
            model,
            "predict_proba",
        ):

            probabilities = (
                model.predict_proba(
                    model_features
                )
            )


            confidence = (

                float(
                    np.max(
                        probabilities[0]
                    )
                )

                * 100
            )


        return (
            str(label),
            confidence,
        )


    # ======================================================
    # VIDEO FRAME
    # ======================================================

    def recv(
        self,
        frame,
    ):


        # ==================================================
        # FRAME → NUMPY
        # ==================================================

        image = frame.to_ndarray(
            format="bgr24"
        )


        # ==================================================
        # MIRROR
        # ==================================================

        image = cv2.flip(
            image,
            1,
        )


        # ==================================================
        # BGR → RGB
        # ==================================================

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )


        # ==================================================
        # MEDIAPIPE
        # ==================================================

        results = self.hands.process(
            rgb
        )


        prediction = "No Hand"

        confidence = 0.0


        # ==================================================
        # HAND FOUND
        # ==================================================

        if results.multi_hand_landmarks:


            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )


            # ------------------------------------------------
            # DRAW LANDMARKS
            # ------------------------------------------------

            mp_drawing.draw_landmarks(

                image,

                hand_landmarks,

                mp_hands.HAND_CONNECTIONS,
            )


            # ------------------------------------------------
            # EXTRACT RAW 63 FEATURES
            # ------------------------------------------------

            features = (
                self.extract_features(
                    hand_landmarks
                )
            )


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            try:

                prediction, confidence = (
                    self.predict(
                        features
                    )
                )


            except Exception as error:

                prediction = (
                    "Prediction Error"
                )

                confidence = 0.0

                print(
                    "Prediction Error:",
                    error
                )


            # ==================================================
            # SMOOTH PREDICTION
            # ==================================================

            if prediction != "Prediction Error":


                self.prediction_history.append(
                    prediction
                )


                common_prediction = (
                    Counter(
                        self.prediction_history
                    ).most_common(1)
                )


                if common_prediction:

                    prediction = (
                        common_prediction[0][0]
                    )


                # ==================================================
                # EXECUTE GESTURE ACTION
                # ==================================================

                if self.mode == (
                    "Gesture Control"
                ):

                    execute_gesture_action(
                        prediction,
                        self,
                    )


        # ==================================================
        # NO HAND
        # ==================================================

        else:

            self.prediction_history.clear()

            prediction = "No Hand"

            confidence = 0.0


        # ==================================================
        # SAVE RESULTS
        # ==================================================

        with self.lock:

            self.current_prediction = (
                prediction
            )

            self.current_confidence = (
                confidence
            )


        # ==================================================
        # TOP INFORMATION PANEL
        # ==================================================

        cv2.rectangle(

            image,

            (
                0,
                0,
            ),

            (
                image.shape[1],
                120,
            ),

            (
                20,
                20,
                20,
            ),

            -1,
        )


        # ==================================================
        # MODE
        # ==================================================

        cv2.putText(

            image,

            self.mode,

            (
                20,
                30,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (
                255,
                255,
                255,
            ),

            2,
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        cv2.putText(

            image,

            f"Prediction: {prediction}",

            (
                20,
                68,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.85,

            (
                0,
                255,
                0,
            ),

            2,
        )


        # ==================================================
        # CONFIDENCE
        # ==================================================

        cv2.putText(

            image,

            f"Confidence: {confidence:.1f}%",

            (
                20,
                103,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (
                0,
                255,
                255,
            ),

            2,
        )


        # ==================================================
        # GESTURE ACTION PANEL
        # ==================================================

        if self.mode == (
            "Gesture Control"
        ):


            cv2.rectangle(

                image,

                (
                    15,
                    130,
                ),

                (
                    image.shape[1] - 15,
                    190,
                ),

                (
                    30,
                    30,
                    30,
                ),

                -1,
            )


            cv2.putText(

                image,

                "Windows Action:",

                (
                    25,
                    155,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (
                    255,
                    255,
                    255,
                ),

                2,
            )


            cv2.putText(

                image,

                self.last_action,

                (
                    25,
                    182,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.70,

                (
                    0,
                    255,
                    0,
                ),

                2,
            )


        # ==================================================
        # RETURN FRAME
        # ==================================================

        return av.VideoFrame.from_ndarray(

            image,

            format="bgr24",
        )


# ==========================================================
# 18. RTC CONFIGURATION
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
# 19. MAIN CAMERA UI
# ==========================================================

if not camera_enabled:

    st.info(
        "📷 Camera is OFF. "
        "Turn on Camera from the sidebar."
    )

else:


    # ======================================================
    # SIGN LANGUAGE UI
    # ======================================================

    if mode == "Sign Language":

        st.subheader(
            "🤟 Sign Language Recognition"
        )

        st.write(
            "Show a trained sign to the camera."
        )


    # ======================================================
    # GESTURE CONTROL UI
    # ======================================================

    else:

        st.subheader(
            "🖱️ Gesture Control"
        )

        st.write(
            "Use trained hand gestures "
            "to control Windows."
        )


    # ======================================================
    # WEBRTC CAMERA
    # ======================================================

    webrtc_streamer(

        key=f"gesturex-camera-{mode}",

        mode=WebRtcMode.SENDRECV,

        rtc_configuration=(
            RTC_CONFIGURATION
        ),

        media_stream_constraints={
            "video": True,
            "audio": False,
        },

        video_processor_factory=(
            GestureXProcessor
        ),

        async_processing=True,
    )


# ==========================================================
# 20. GESTURE CONTROL INFORMATION
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
            **✋ Palm** → Palm Detected

            **✊ Fist** → Play / Pause

            **👍 Thumbs Up** → Volume Up

            **👎 Thumbs Down** → Volume Down

            **☝️ Index Right** → Right Arrow
            """
        )


    with col2:

        st.markdown(
            """
            **☝️ Index Left** → Left Arrow

            **⬆️ Up** → Up Arrow

            **⬇️ Down** → Down Arrow

            **👌 OK** → Enter

            **No Gesture** → No Action
            """
        )


# ==========================================================
# 21. SIGN LANGUAGE INFORMATION
# ==========================================================

else:

    st.markdown("---")

    st.subheader(
        "🤟 Sign Language"
    )

    st.success(
        "Sign Language Random Forest model "
        "is running with its original "
        "wrist and hand-size normalization."
    )


# ==========================================================
# 22. LOADED MODEL PATHS
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

    if gesture_scaler is not None:

        st.code(
            GESTURE_SCALER_FILE
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

    st.info(
        "Sign Language does not use "
        "scaler.pkl. It uses the original "
        "wrist + hand-size normalization."
    )


# ==========================================================
# 23. FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#888;">
    GestureX — AI Sign Language & Gesture Control System
    </div>
    """,
    unsafe_allow_html=True,
)