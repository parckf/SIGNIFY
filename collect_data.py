import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Initialize MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    num_hands=1
)

# Create dataset file
file_exists = os.path.exists("asl_dataset.csv")

file = open("asl_dataset.csv", "a", newline="")
writer = csv.writer(file)

if not file_exists:
    writer.writerow(
        ["label"] +
        [f"feature_{i}" for i in range(63)]
    )

# Start camera
camera = cv2.VideoCapture(0)

print("SIGNIFY DATA COLLECTION STARTED")
print("Show a sign and press its corresponding letter.")
print("Press Q to quit.")

try:
    with HandLandmarker.create_from_options(options) as detector:

        while True:

            success, frame = camera.read()

            if not success:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                # Normalize landmarks relative to wrist
                wrist = hand[0]

                coordinates = []

                for landmark in hand:
                    coordinates.extend([
                        landmark.x - wrist.x,
                        landmark.y - wrist.y,
                        landmark.z - wrist.z
                    ])

                # Scale coordinates
                max_value = max(
                    abs(value) for value in coordinates
                )

                if max_value > 0:
                    coordinates = [
                        value / max_value
                        for value in coordinates
                    ]

                # Draw hand landmarks
                height, width, _ = frame.shape

                for landmark in hand:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                cv2.putText(
                    frame,
                    "HAND DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:
                coordinates = None

                cv2.putText(
                    frame,
                    "SHOW YOUR HAND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("SIGNIFY DATA COLLECTION", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('9'):
                break

            # Save sample when a letter is pressed
            if (
                coordinates is not None
                and ord('a') <= key <= ord('z')
            ):

                label = chr(key)

                writer.writerow(
                    [label] + coordinates
                )

                file.flush()

                print(f"Saved sample for: {label.upper()}")

finally:
    camera.release()
    file.close()
    cv2.destroyAllWindows()