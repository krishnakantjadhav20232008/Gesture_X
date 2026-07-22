import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import cv2

# Open default webcam with DirectShow backend for Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started successfully.")
print("Press 'q' to quit.")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
        
    # 1. Display the frame in a window
    cv2.imshow("SignNova - Camera Test", frame)
    
    # 2. Wait 1 millisecond for a keypress; exit loop if 'e' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 3. Clean up and close windows properly on exit
cap.release()
cv2.destroyAllWindows()