"""
GestureX - Main Application Launcher

Run from the project root with:
    python main.py

This launcher starts the Streamlit application located at:
    ui/app.py
"""

import os
import sys
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
APP_FILE = PROJECT_ROOT / "ui" / "app.py"


def main():
    """Launch the GestureX Streamlit application."""

    if not APP_FILE.exists():
        print("ERROR: Streamlit application not found.")
        print(f"Expected file: {APP_FILE}")
        sys.exit(1)

    print("=" * 60)
    print("        GestureX - AI Sign Language & Gesture Control")
    print("=" * 60)
    print(f"Project Root : {PROJECT_ROOT}")
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
        print("    pip install -r requirements.txt")
        sys.exit(1)

    except subprocess.CalledProcessError as error:
        print(
            f"\nERROR: GestureX stopped with exit code "
            f"{error.returncode}."
        )
        sys.exit(error.returncode)

    except KeyboardInterrupt:
        print("\nGestureX stopped by user.")


if __name__ == "__main__":
    main()