# GestureX 🖐️

AI Sign Language & Gesture Control System using Computer Vision and Machine Learning.

## Gesture Control

| Gesture | Windows Action |
|---|---|
| ✊ Fist | Play / Pause |
| 👍 Thumbs Up | Volume Up |
| 👎 Thumbs Down | Volume Down |
| ☝️ Index Right | Right Arrow |
| ☝️ Index Left | Left Arrow |
| 👌 OK | Enter |
| ✋ Palm | Palm Detected |
| No Gesture | No Action |

## Sign Language Recognition

GestureX uses a trained Random Forest model with MediaPipe hand landmarks to recognize sign-language characters.

## Technology Stack

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- Random Forest
- Streamlit
- Streamlit-WebRTC
- NumPy
- Joblib

## Main Features

- Real-time hand detection
- 21 MediaPipe hand landmarks
- 63 landmark features
- Sign-language recognition
- Real-time gesture classification
- Windows keyboard/media control
- Volume control
- Play/Pause control
- Arrow-key control
- Enter-key control
- Interactive Streamlit interface

## Project Structure

```text
AI_Sign_Gesture_System/
│
├── ui/
│   └── app.py
│
├── outputs/
│   ├── models_signs/
│   ├── model_gesture/
│   ├── preprocessed_signs/
│   └── preprocessed_gesture/
│
├── README.md
└── ...
```

## Running the Project

Open the project folder in VS Code and activate your Python environment.

Install the required packages:

```powershell
pip install -r requirements.txt
```

Run the Streamlit application:

```powershell
streamlit run ui/app.py
```

Then open the Streamlit URL shown in the terminal.

## Project Status

✅ Sign Language Recognition  
✅ Gesture Control  
✅ Real-time Hand Detection  
✅ Windows Gesture Actions  
✅ Streamlit UI

## Author

**Krishnakant Jadhav**

AI Sign Language & Gesture Control System
