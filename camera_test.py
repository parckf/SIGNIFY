import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    exit()

print("SIGNIFY CAMERA STARTED")
print("Press Q or ESC to exit")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Camera frame unavailable")
            break

        cv2.imshow("SIGNIFY - Live Camera", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == ord('q') or key == 27:
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Camera closed successfully!")