import cv2
import mediapipe as mp


# ==========================================================
# MEDIAPIPE SETUP
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
# WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)

print("\n===================================")
print("REAL-TIME HAND TRACKING")
print("===================================")
print("Press Q to exit.")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to access webcam.")
        break


    # Flip for natural interaction
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Detect hands
    results = hands.process(
        rgb_frame
    )


    # ======================================================
    # DRAW LANDMARKS
    # ======================================================

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


    # ======================================================
    # DISPLAY
    # ======================================================

    cv2.imshow(
        "GestureX - Hand Tracking",
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================================
# CLEANUP
# ==========================================================

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\nHand tracking stopped.")