# 🖐️ GestureX

## AI Sign Language Recognition & Gesture-Based Windows Control System

> **GestureX** is a real-time computer vision and machine learning system that recognizes hand signs and gestures through a webcam. It provides two major functionalities: **Sign Language Recognition** and **Gesture-Based Windows Control**.

---

## 📌 Project Overview

GestureX is an AI-powered system developed using **Computer Vision, MediaPipe Hand Tracking, Machine Learning, and Streamlit**.

The system captures live video from a webcam, detects the user's hand, extracts hand-landmark features, and uses trained **Random Forest classification models** to recognize either sign-language alphabets or predefined computer-control gestures.

GestureX operates in two modes:

### 🤟 1. Sign Language Recognition

The system recognizes trained hand signs representing alphabets and displays the predicted character with its confidence score.

### 🖱️ 2. Gesture Control

The system recognizes predefined hand gestures and converts them into Windows keyboard/media actions such as:

* Volume Up
* Volume Down
* Play/Pause
* Left Arrow
* Right Arrow
* Up Arrow
* Down Arrow
* Enter/OK

---

# 🎯 Objectives

The main objectives of GestureX are:

1. To develop a real-time hand gesture recognition system.
2. To recognize sign-language alphabets using machine learning.
3. To provide gesture-based computer interaction.
4. To reduce dependence on traditional input devices for selected tasks.
5. To demonstrate the practical application of computer vision and machine learning.
6. To create an easy-to-use graphical interface for real-time interaction.
7. To integrate hand tracking, machine learning prediction, and Windows control into a single system.

---

# 🚨 Problem Statement

Traditional computer interaction mainly depends on devices such as keyboards, mice, and touchscreens.

People with limited access to traditional input methods may benefit from alternative interaction techniques. Sign language also provides an important communication method, but computers cannot naturally understand hand signs without specialized recognition systems.

GestureX addresses this problem by providing a system capable of:

**Hand → Landmark Detection → Feature Extraction → Machine Learning → Sign/Gesture Recognition → Output/Computer Action**

---

# 💡 Proposed Solution

GestureX uses a webcam to capture the user's hand in real time.

The captured frame is processed using **MediaPipe Hands**, which detects 21 hand landmarks.

Each landmark contains:

* X coordinate
* Y coordinate
* Z coordinate

Therefore:

**21 landmarks × 3 coordinates = 63 features**

These features are provided to trained Random Forest models.

Depending on the selected mode:

```text
                    Webcam
                       │
                       ▼
              MediaPipe Hands
                       │
                       ▼
             21 Hand Landmarks
                       │
                       ▼
              63 Feature Values
                       │
                       ▼
              Random Forest Model
                    /       \
                   /         \
                  ▼           ▼
        Sign Language     Gesture Control
             │                  │
             ▼                  ▼
        A-Z Prediction     Windows Action
```

---

# 🧠 System Architecture

The overall architecture consists of the following components:

### 1. Input Layer

A webcam captures real-time hand images.

### 2. Hand Detection Layer

MediaPipe Hands detects the user's hand and identifies 21 hand landmarks.

### 3. Feature Extraction Layer

The X, Y, and Z coordinates of the 21 landmarks are extracted.

This produces 63 numerical features.

### 4. Machine Learning Layer

Random Forest classification models predict:

* Sign-language classes
* Gesture-control classes

### 5. Output Layer

Depending on the selected mode:

**Sign Language Mode**

```text
Hand → Sign → Predicted Character
```

**Gesture Control Mode**

```text
Hand → Gesture → Windows Action
```

### 6. User Interface

A Streamlit-based interface provides:

* Mode selection
* Camera control
* Real-time prediction
* Confidence information
* Gesture action information
* Model information

---

# 🤟 Sign Language Recognition

The Sign Language Recognition module is designed to recognize trained hand signs.

The system loads:

```text
Random Forest Model
Label Encoder
Scaler (if available)
```

The model predicts the encoded class, which is then converted back into the corresponding human-readable label using the label encoder.

Example:

```text
Hand Sign
   ↓
63 Features
   ↓
Random Forest
   ↓
Encoded Prediction
   ↓
Label Encoder
   ↓
Character
```

The current trained model contains:

**[EDIT: ENTER YOUR EXACT NUMBER OF SIGN CLASSES]**

Example:

```text
A, B, C, D, ... Z
```

---

# 🎮 Gesture Control

GestureX also provides real-time computer control through predefined gestures.

| Gesture         | Windows Action |
| --------------- | -------------- |
| ✊ Fist          | Play / Pause   |
| 👍 Thumbs Up    | Volume Up      |
| 👎 Thumbs Down  | Volume Down    |
| ☝️ Index Right  | Right Arrow    |
| ☝️ Index Left   | Left Arrow     |
| ⬆️ Up Gesture   | Up Arrow       |
| ⬇️ Down Gesture | Down Arrow     |
| 👌 OK           | Enter          |
| ✋ Palm          | Palm Detected  |
| No Gesture      | No Action      |

The Windows control system uses Windows virtual-key codes to simulate keyboard and media-key actions.

---

# 🤖 Machine Learning

GestureX uses **Random Forest Classification** for gesture and sign prediction.

Random Forest was selected because it:

* Works well with structured numerical features
* Handles nonlinear classification problems
* Provides good classification performance
* Supports multiple classes
* Provides prediction probabilities
* Is relatively efficient for real-time inference

---

# ✋ Hand Landmark Detection

GestureX uses **MediaPipe Hands** for hand tracking.

The system detects up to one hand at a time in the current implementation.

Each detected hand provides 21 landmarks.

Each landmark contains:

```text
X
Y
Z
```

Therefore:

```text
21 × 3 = 63 features
```

These 63 features are used as input to the machine learning model.

---

# 📊 Feature Processing

The feature-processing pipeline includes:

1. Hand detection
2. Landmark extraction
3. Conversion to numerical arrays
4. Feature reshaping
5. Optional scaling
6. Machine learning prediction

Where a scaler is available, the feature vector is transformed before prediction.

This ensures that the preprocessing used during inference is consistent with the trained model.

---

# 🔄 Prediction Smoothing

Real-time webcam predictions can sometimes fluctuate between consecutive frames.

To improve stability, GestureX stores recent predictions in a small history buffer.

The most frequently occurring prediction is selected as the current prediction.

Conceptually:

```text
Frame 1 → A
Frame 2 → A
Frame 3 → B
Frame 4 → A
Frame 5 → A

Final Prediction → A
```

This reduces random frame-to-frame prediction changes.

---

# 📈 Confidence Score

When supported by the Random Forest model, GestureX uses `predict_proba()` to calculate the highest class probability.

The confidence is displayed as a percentage.

Example:

```text
Prediction: A
Confidence: 96.4%
```

> **Note:** Confidence is the model's predicted probability, not a guarantee that the recognition is correct.

---

# 🖥️ User Interface

GestureX uses **Streamlit** for the graphical user interface.

The interface provides:

### Sidebar

* Mode selection
* Camera enable/disable
* Model information
* Number of classes
* Scaler information

### Main Area

* Webcam stream
* Hand landmarks
* Current prediction
* Confidence
* Windows action

---

# 🛠️ Technologies Used

| Technology       | Purpose                    |
| ---------------- | -------------------------- |
| Python           | Main programming language  |
| OpenCV           | Image and video processing |
| MediaPipe        | Hand landmark detection    |
| NumPy            | Numerical processing       |
| Scikit-learn     | Machine learning           |
| Random Forest    | Classification             |
| Joblib           | Model serialization        |
| Streamlit        | User interface             |
| Streamlit-WebRTC | Real-time webcam streaming |
| PyAV             | Video frame processing     |
| Windows API      | Keyboard/media control     |

---

# 📂 Project Structure

The project is organized into separate modules for better maintainability.

```text
AI_Sign_Gesture_System-
│
├── README.md
├── requirements.txt
│
├── docs/
│   ├── Product Requirements Document
│   └── System Architecture
│
├── models/
│
├── outputs/
│   ├── model_gesture/
│   ├── preprocessed_gesture/
│   ├── models_signs/
│   └── preprocessed_signs/
│
├── preprocessed_gesture/
│
├── preprocessed_signs/
│
└── src/
    │
    ├── gesture_control/
    │
    ├── hand_tracking/
    │
    ├── sign_recognition/
    │
    └── ui/
        └── app.py
```

> **[EDIT THIS TREE IF YOUR CURRENT GITHUB STRUCTURE DIFFERS.]**

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/krishnakantjadhav20232008/AI_Sign_Gesture_System-.git
```

Move into the project directory:

```bash
cd AI_Sign_Gesture_System-
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running GestureX

The main Streamlit application is located inside the UI module.

Run:

```bash
streamlit run src/ui/app.py
```

After starting Streamlit, open the displayed local URL in your browser.

Typical Streamlit URL:

```text
http://localhost:8501
```

> **[EDIT IF YOUR ACTUAL RUN COMMAND IS DIFFERENT.]**

---

# 🎥 How to Use

## Sign Language Mode

1. Start GestureX.
2. Enable the camera.
3. Select **Sign Language** mode.
4. Place your hand in front of the camera.
5. Show a trained sign.
6. GestureX detects the hand landmarks.
7. The Random Forest model predicts the sign.
8. The predicted character and confidence are displayed.

---

## Gesture Control Mode

1. Start GestureX.
2. Enable the camera.
3. Select **Gesture Control** mode.
4. Show a trained gesture.
5. GestureX predicts the gesture.
6. The corresponding Windows action is executed.

Example:

```text
👍
 ↓
Thumbs Up
 ↓
Windows Volume Up
```

---

# 🧪 Testing

The system was tested using real-time webcam input.

Testing included:

### Sign Language

* Hand detection
* Landmark extraction
* Sign prediction
* Confidence calculation
* Prediction smoothing

### Gesture Control

* Gesture detection
* Windows keyboard control
* Media-key control
* Action cooldown
* Real-time response

> **[EDIT: ADD YOUR ACTUAL TEST RESULTS/PERCENTAGES HERE.]**

---

# 📊 Results

GestureX successfully integrates real-time hand tracking with machine learning-based recognition.

The completed system provides:

* Real-time hand detection
* 63-feature landmark extraction
* Sign-language recognition
* Gesture recognition
* Windows computer control
* Real-time confidence display
* Interactive Streamlit interface

### Model Performance

| Model                         |    Accuracy |
| ----------------------------- | ----------: |
| Sign Language Random Forest   | **100 %** |
| Gesture Control Random Forest | **100 %** |

# ⭐ Key Features

* 🤟 Real-time sign-language recognition
* 🖐️ MediaPipe hand tracking
* 🔢 63-dimensional landmark feature extraction
* 🤖 Random Forest classification
* 🎮 Gesture-based Windows control
* 🔊 Volume control
* ⏯️ Media play/pause
* ⬅️➡️ Arrow-key control
* ⏎ Enter control
* 📊 Confidence estimation
* 🔄 Prediction smoothing
* 📷 Real-time webcam processing
* 🖥️ Streamlit user interface
* 🧩 Modular project architecture

---

# 🔐 Safety and Privacy

GestureX processes webcam frames for real-time recognition.

The application should be used with appropriate camera permissions.

The project does not require uploading webcam frames to an external cloud service for the core recognition pipeline.

> **[EDIT THIS SECTION IF YOUR IMPLEMENTATION USES ANY EXTERNAL CLOUD/API SERVICE.]**

---

# ⚠️ Limitations

Current limitations may include:

1. Recognition depends on the quality of the trained dataset.
2. Poor lighting can affect hand detection.
3. Occluded or partially visible hands may reduce accuracy.
4. Extreme hand orientations may produce incorrect predictions.
5. Background clutter can affect detection.
6. Real-time performance depends on hardware.
7. The current system is trained only for the gesture/sign classes included in the dataset.
8. Gesture control is currently designed for Windows.
9. The system currently processes one hand at a time.

---

# 🚀 Future Scope

Future improvements may include:

### 1. Expanded Sign Language Vocabulary

Support more signs, words, phrases, and complete sentences.

### 2. Two-Hand Recognition

Support simultaneous recognition of both hands.

### 3. Dynamic Gesture Recognition

Recognize gestures that involve hand movement over time.

### 4. Voice Output

Convert recognized signs into speech.

### 5. Multi-Platform Control

Extend computer-control functionality to Linux and macOS.

### 6. Deep Learning

Explore CNN, LSTM, Transformer, or other deep-learning architectures.

### 7. Mobile Application

Develop an Android/iOS version.

### 8. Improved UI

Add richer visualizations, gesture history, statistics, and customization.

### 9. Personalized Gesture Profiles

Allow users to train and configure their own gestures.

### 10. Accessibility Applications

Use GestureX for assistive computer interaction and communication systems.

---

# 🎓 Applications

GestureX can potentially be used in:

* Accessibility systems
* Human-computer interaction
* Educational applications
* Sign-language learning
* Assistive technology
* Smart computer interfaces
* Touchless control systems
* Interactive demonstrations
* Research projects

---

# 🧩 Development Workflow

The development process followed these major stages:

```text
Requirement Analysis
        ↓
Project Planning
        ↓
Dataset Preparation
        ↓
Hand Landmark Extraction
        ↓
Feature Preprocessing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Real-Time Prediction
        ↓
Gesture Control Integration
        ↓
Streamlit UI Development
        ↓
Testing & Debugging
        ↓
Final System
```

---

# 🧠 Learning Outcomes

Through the development of GestureX, the project demonstrates practical knowledge of:

* Python programming
* Machine learning
* Computer vision
* Hand landmark detection
* Feature engineering
* Random Forest classification
* Model preprocessing
* Real-time video processing
* Streamlit application development
* WebRTC-based video streaming
* Windows API interaction
* Software project organization

---

# 📚 References

Add the resources used during development.

Suggested references:

1. MediaPipe Hands documentation
2. OpenCV documentation
3. Scikit-learn documentation
4. Streamlit documentation
5. Streamlit-WebRTC documentation
6. Python documentation
---

# 👨‍💻 Developer

**Name:** Krishnakant Pawan Jadhav & Hanzala Zahid Ahmad 

**Project:** GestureX – AI Sign Language Recognition & Gesture-Based Windows Control System

**Institution:** Maulana Mukthar Ahmad Nadvi Technical Campus

**Department:** Computer 

**Internship:** Sumago Infotec Pvt. Ltd

**Mentor:** Payal Patil 

---

# 📄 Project Documentation

Detailed project documentation is available in the `docs/` directory.

The documentation includes:

* Product Requirements Document
* System Architecture
* Project planning
* Technical design

Additional documentation and final project reports can be added as the project progresses.

---

# 📜 License

This project is created for **educational, research, and internship purposes**.

> **[ADD YOUR CHOSEN LICENSE HERE, e.g. MIT License, if applicable.]**

---

# 🙌 Acknowledgements

Special thanks to:

* Payal Patil
* IMMANTC Malegaon
* Hanzala Zahid Ahmad 
* Open-source communities
* MediaPipe
* OpenCV
* Scikit-learn
* Streamlit

---

# ⭐ Final Note

GestureX demonstrates how **Artificial Intelligence + Computer Vision + Machine Learning** can be combined to create a practical real-time human-computer interaction system.

The project brings together:

```text
🤟 Sign Language Recognition
            +
🖐️ Hand Tracking
            +
🤖 Machine Learning
            +
🎮 Gesture Control
            +
🖥️ Interactive UI
            =
🚀 GestureX
```

**Built with Python, Computer Vision, and Machine Learning.**

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.

**GestureX — Turning Hand Gestures into Digital Interaction.**