import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
from matplotlib import pyplot as plt
from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

def display_one_image(image, title, titlesize=16, title_color=(255, 255, 255), bg_color=(1, 1, 1)):
    font_scale = titlesize / 10
    thickness = int(titlesize / 10)
    text_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_width, text_height = text_size

    title_region_height = text_height + 30
    annotated_image = cv2.copyMakeBorder(image, top=title_region_height, bottom=0, left=0, right=0,
                                         borderType=cv2.BORDER_CONSTANT, value=bg_color)

    text_x = 10
    text_y = title_region_height - 10

    cv2.putText(annotated_image, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, title_color, thickness)

    return annotated_image

def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows
    FIGSIZE = 13.0
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

    result_images = []

    for i, (image, gestures) in enumerate(zip(images[:rows * cols], gestures[:rows * cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        result_image = display_one_image(annotated_image, title)
        result_images.append(result_image)
    return result_images

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

video_path = 'C:\\Users\\hegyi\\OneDrive\\Pictures\\Camera Roll\\WIN_20240227_19_34_39_Pro.mp4'
frames = 0
elapsed = 0
cap = cv2.VideoCapture(video_path)
imageBatch = []
resultsBatch = []

start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = recognizer.recognize(image)
    if result.gestures:
        top_gesture = result.gestures[0][0]
        hand_landmarks = result.hand_landmarks

        imageBatch.append(image)
        resultsBatch.append((top_gesture, hand_landmarks))

    frames += 1
end_time = time.time()
duration = end_time - start_time
fps = frames / duration
print(f"FPS: {fps}")
print(f"Processed frames in: {duration} seconds")
processed_images = display_batch_of_images_with_gestures_and_hand_landmarks(imageBatch, resultsBatch)

frame_height, frame_width, _ = processed_images[0].shape
out = cv2.VideoWriter('result_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

for img in processed_images:
    out.write(img)
out.release()
cap.release()
cv2.destroyAllWindows()
