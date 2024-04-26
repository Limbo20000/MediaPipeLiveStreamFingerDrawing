# Task 3.6
import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

video = cv2.VideoCapture(0)

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE)

while True:
    ret, frame = video.read()
    if not ret:
        print("No Frame to capture")
        break

    cv2.imshow('Gesture Recognition by Image', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        with GestureRecognizer.create_from_options(options) as recognizer:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = recognizer.recognize(mp_image)

            if result.gestures:
                for gesture in result.gestures:
                    print(f"Category: {gesture[0].category_name}" + "\n" + f"Accuracy: {gesture[0].score:.2f}")

    elif cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()