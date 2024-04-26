#Task 3.8
import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

video = cv2.VideoCapture(0)
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    frame = output_image.numpy_view()  # Convert to numpy array

    if result.gestures:
        # Access recognized gestures
        gestures = result.gestures[0]
        # print(gestures[0].category_name)
        # print(gestures[0].score)
        print(f"Category: {gestures[0].category_name}\n")
        print(f"Confidence Score: {gestures[0].score:.2f}")


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

timestamp = 0
with GestureRecognizer.create_from_options(options) as recognizer:
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            print("Ignoring empty frame")
            break

        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognizer.recognize_async(mp_image, timestamp)
        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

video.release()
cv2.destroyAllWindows()
