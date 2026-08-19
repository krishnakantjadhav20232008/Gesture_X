"""
GestureX - Main Application Launcher

Run from the project root with:
    python main.py
"""

import os
import sys
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
UI_DIR = PROJECT_ROOT / "ui"
APP_FILE = UI_DIR / "app.py"
MODEL_DIR = PROJECT_ROOT / "models" / "outputs"


def ensure_ui_app_exists():
    """Create a default app.py if ui/app.py is missing."""
    UI_DIR.mkdir(parents=True, exist_ok=True)

    if not APP_FILE.exists():
        default_app_code = f'''import streamlit as st
import pickle
import os
from pathlib import Path

st.set_page_config(page_title="GestureX Control", layout="wide")

st.title(" GestureX - AI Sign Language & Gesture Control")

MODEL_DIR = Path(r"{MODEL_DIR}")

@st.cache_resource
def load_models():
    model_path = MODEL_DIR / "gesture_control_rf.pkl"
    encoder_path = MODEL_DIR / "gesture_encoder.pkl"
    
    # Fallback to alternate file names if primary aren't present
    if not model_path.exists():
        model_path = MODEL_DIR / "random_forest.pkl"
    if not encoder_path.exists():
        encoder_path = MODEL_DIR / "label_encoder.pkl"
        
    if not model_path.exists() or not encoder_path.exists():
        return None, None
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
        
    return model, encoder

model, encoder = load_models()

if model and encoder:
    st.success(" ML Models Loaded Successfully from `models/outputs/`!")
    st.sidebar.header("System Status")
    st.sidebar.write("Model Loaded: Ready")
    st.write("### Model Inspection")
    st.write(f"**Model Type:** {{type(model).__name__}}")
    if hasattr(encoder, 'classes_'):
        st.write(f"**Recognized Classes:** {{list(encoder.classes_)}}")
else:
    st.error(" Model or Encoder missing in `models/outputs/`. Please verify your files.")
'''
        with open(APP_FILE, "w", encoding="utf-8") as f:
            f.write(default_app_code)


def main():
    """Launch the GestureX Streamlit application."""

    # Ensure app.py exists
    ensure_ui_app_exists()

    print("=" * 60)
    print("        GestureX - AI Sign Language & Gesture Control")
    print("=" * 60)
    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Model Dir    : {MODEL_DIR}")
    print(f"Application  : {APP_FILE}")
    print()
    print("Starting GestureX...")
    print("The Streamlit browser window will open shortly.")
    print("Press Ctrl+C in this terminal to stop the application.")
    print("=" * 60)

    # Start Streamlit using the same Python environment
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.headless=false",
    ]

    try:
        subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            check=True,
        )

    except FileNotFoundError:
        print("\nERROR: Streamlit is not installed.")
        print("Install project requirements with:")
        print("    pip install streamlit")
        sys.exit(1)

    except subprocess.CalledProcessError as error:
        print(f"\nERROR: GestureX stopped with exit code {error.returncode}.")
        sys.exit(error.returncode)

    except KeyboardInterrupt:
        print("\nGestureX stopped by user.")


if __name__ == "__main__":
    main()