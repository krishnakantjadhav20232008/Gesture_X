import os
import cv2
import csv
import mediapipe as mp
from collections import defaultdict

# ==========================================================
# PATHS
# ==========================================================

DATASET_PATH = r"D:\AI_Datasets"

CSV_FILE = r"D:\AI_Sign_Gesture_System\outputs\gesturex_dataset.csv"

os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)


# ==========================================================
# MEDIAPIPE
# ==========================================================

mp_hands = mp.solutions.hands


# ==========================================================
# IMAGE EXTENSIONS
# ==========================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ==========================================================
# CSV HEADER
# ==========================================================

header = []

for i in range(21):
    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

header.append("label")


# ==========================================================
# COUNTERS
# ==========================================================

total_images = 0
successful = 0
failed = 0


# ==========================================================
# FUNCTION: EXTRACT LANDMARKS
# ==========================================================

def extract_landmarks(image, hands):

    # Try original image
    images_to_try = []

    images_to_try.append(image)

    # Resize
    resized = cv2.resize(
        image,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    images_to_try.append(resized)

    # Contrast enhancement
    lab = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    images_to_try.append(enhanced)

    # Flipped version
    flipped = cv2.flip(
        enhanced,
        1
    )

    images_to_try.append(flipped)


    # ------------------------------------------------------
    # TRY EACH VERSION
    # ------------------------------------------------------

    for processed_image in images_to_try:

        rgb = cv2.cvtColor(
            processed_image,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            # First detected hand
            hand = results.multi_hand_landmarks[0]

            row = []

            for landmark in hand.landmark:

                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            return row


    # ------------------------------------------------------
    # NO LANDMARK FOUND
    # ------------------------------------------------------

    return None


# ==========================================================
# CREATE CSV
# ==========================================================

with open(
    CSV_FILE,
    mode="w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(header)


    # ======================================================
    # MEDIAPIPE HANDS
    # ======================================================

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3
    ) as hands:


        # ==================================================
        # DATASETS
        # ==================================================

        for dataset_name in sorted(
            os.listdir(DATASET_PATH)
        ):

            dataset_folder = os.path.join(
                DATASET_PATH,
                dataset_name
            )

            if not os.path.isdir(
                dataset_folder
            ):
                continue


            print("\n" + "=" * 60)
            print(
                f"DATASET: {dataset_name}"
            )
            print("=" * 60)


            # ==================================================
            # CLASSES
            # ==================================================

            for label in sorted(
                os.listdir(dataset_folder)
            ):

                label_folder = os.path.join(
                    dataset_folder,
                    label
                )

                if not os.path.isdir(
                    label_folder
                ):
                    continue


                class_total = 0
                class_success = 0
                class_failed = 0


                print(
                    f"\nClass: {label}"
                )


                # ==================================================
                # IMAGES
                # ==================================================

                for image_name in sorted(
                    os.listdir(label_folder)
                ):

                    if not image_name.lower().endswith(
                        IMAGE_EXTENSIONS
                    ):
                        continue


                    image_path = os.path.join(
                        label_folder,
                        image_name
                    )

                    total_images += 1
                    class_total += 1


                    # ------------------------------------------------
                    # READ IMAGE
                    # ------------------------------------------------

                    image = cv2.imread(
                        image_path
                    )


                    if image is None:

                        failed += 1
                        class_failed += 1

                        print(
                            f"[FAILED READ] "
                            f"{image_name}"
                        )

                        continue


                    # ------------------------------------------------
                    # EXTRACT
                    # ------------------------------------------------

                    landmarks = extract_landmarks(
                        image,
                        hands
                    )


                    # ------------------------------------------------
                    # SUCCESS
                    # ------------------------------------------------

                    if landmarks is not None:

                        landmarks.append(
                            label
                        )

                        writer.writerow(
                            landmarks
                        )

                        successful += 1
                        class_success += 1


                    # ------------------------------------------------
                    # FAILURE
                    # ------------------------------------------------

                    else:

                        failed += 1
                        class_failed += 1

                        print(
                            f"[NO LANDMARK] "
                            f"{image_name}"
                        )


                # ==================================================
                # CLASS SUMMARY
                # ==================================================

                print(
                    f"Total: {class_total} | "
                    f"Extracted: {class_success} | "
                    f"Failed: {class_failed}"
                )


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("LANDMARK EXTRACTION COMPLETED")
print("=" * 60)

print(
    f"Total Images : {total_images}"
)

print(
    f"Successful   : {successful}"
)

print(
    f"Failed       : {failed}"
)

print(
    f"\nCSV Saved At:"
)

print(CSV_FILE)

print("=" * 60)