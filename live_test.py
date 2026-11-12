import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load our trained AI model
model = joblib.load("signify_model.pkl")

# Initialize MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    num_hands=1
)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam!")
    exit()

print("SIGNIFY LIVE RECOGNITION STARTED")
print("Press Q to quit.")

try:
    with HandLandmarker.create_from_options(options) as detector:

        while True:
            success, frame = camera.read()

            if not success:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                # Extract landmarks
                wrist = hand[0]
                coordinates = []

                for landmark in hand:
                    coordinates.extend([
                        landmark.x - wrist.x,
                        landmark.y - wrist.y,
                        landmark.z - wrist.z
                    ])

                # Normalize coordinates
                max_value = max(abs(v) for v in coordinates)

                if max_value > 0:
                    coordinates = [
                        v / max_value for v in coordinates
                    ]

                    # Ask AI to predict the sign
                    prediction = model.predict([coordinates])[0]

                    probabilities = model.predict_proba(
                        [coordinates]
                    )[0]

                    confidence = max(probabilities) * 100

                    # Display prediction
                    cv2.putText(
                        frame,
                        f"ASL: {prediction.upper()}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        frame,
                        f"Confidence: {confidence:.1f}%",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                # Draw landmarks
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

            else:
                cv2.putText(
                    frame,
                    "SHOW YOUR HAND",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("SIGNIFY - LIVE ASL RECOGNITION", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()