import os

DATASET_PATH = r"D:\AI_Datasets"

print("\nImage Count Per Class\n")

for dataset in os.listdir(DATASET_PATH):

    dataset_path = os.path.join(DATASET_PATH, dataset)

    if not os.path.isdir(dataset_path):
        continue

    print(f"\nDataset: {dataset}")

    for label in sorted(os.listdir(dataset_path)):

        label_path = os.path.join(dataset_path, label)

        if not os.path.isdir(label_path):
            continue

        image_count = 0

        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)

            if os.path.isfile(image_path):
                image_count += 1

        print(f"{label}: {image_count}")