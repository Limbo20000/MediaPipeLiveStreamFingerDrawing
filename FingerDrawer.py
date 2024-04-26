import cv2
import mediapipe as mp
import numpy as np
import time


i = 0

class FingerTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.prev_x, self.prev_y = None, None
        self.scribble = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
        self.scribble_index = 0  # Initialize scribble index

    def track_finger_movement(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Assuming only one hand is present

            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            if self.prev_x is not None and self.prev_y is not None:
                cv2.line(self.scribble, (self.prev_x, self.prev_y), (x, y), (0, 0, 255), 2)

            self.prev_x, self.prev_y = x, y

        return frame

    def save_scribble(self, i=None):
        filename = 'scribble' + str(i) + '.png'
        cv2.imwrite(filename, self.scribble)
        self.scribble_index += 1

    def clear_scribble(self):
        self.scribble = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Reset to white background

def main():
    cap = cv2.VideoCapture(0)
    finger_tracker = FingerTracker()

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = finger_tracker.track_finger_movement(frame)
        cv2.imshow('Scribble', finger_tracker.scribble)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('s'):
            finger_tracker.save_scribble(i)
        if cv2.waitKey(1) == ord('q'):
            finger_tracker.clear_scribble()

        frame_count += 1

        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            start_time = time.time()
            frame_count = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
