#create an object called classifier which will take the current frame as an instantiation parameter
#then it will use that frame and 2 functions to figure out the likely gesture and it's confidence

#how will this object be used?
#at the start of the main program, a new instance of this object will be called
#at every frame, the attributes of the object will be updated and then a function will be called to return the gesture and it's confidence

import pickle
import cv2
import mediapipe as mp
import numpy as np


class Classifier:
    
    def __init__(self):
        self.frame = None
        self.results = None
        self.prediction = None
        self.predicted_gesture = None
        self.confidence = None

        self.data_aux = []
        self.x_ = []
        self.y_ = []

        self.frame_rgb = None

        self.model_file = './models/model3.pickle'
        self.model_dict = pickle.load(open(self.model_file, 'rb'))
        self.model = self.model_dict['model']

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)
        self.labels_dict = {0: "closed", 1: "previous", 2: "next", 3: "pointer", 4: "drawer", 5: "erase"}

        self.ESC_key = 27
        self.y_wrist = 0
        self.pointer_coord = (0,0)

    def reset_results(self):
        self.results = None
        self.prediction = None
        self.predicted_gesture = None
        self.confidence = None
        self.data_aux = []
        self.x_ = []
        self.y_ = []

    def new_frame(self, frame):
        self.reset_results()
        self.frame = frame
        H, W, _ = frame.shape
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(self.frame_rgb)

        if self.results.multi_hand_landmarks:
            max_trans_x = float("-inf")
            min_trans_x = float("inf")

            max_trans_y = float("-inf")
            min_trans_y = float("inf")

            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                self.frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
                )
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    trans_x = x - hand_landmarks.landmark[0].x
                    trans_y = y - hand_landmarks.landmark[0].y

                    max_trans_x = max(max_trans_x, trans_x)
                    min_trans_x = min(min_trans_x, trans_x)

                    max_trans_y = max(max_trans_y, trans_y)
                    min_trans_y = min(min_trans_y, trans_y)

                    self.data_aux.append(trans_x)
                    self.data_aux.append(trans_y)
                    self.x_.append(x)
                    self.y_.append(y)
            
            x_range = max_trans_x - min_trans_x
            y_range = max_trans_y - min_trans_y
            handsize = max(x_range, y_range)

            for i in range(len(self.data_aux)):
                self.data_aux[i] /= handsize

            first_hand = self.results.multi_hand_landmarks[0]
            landmarks = first_hand.landmark

            #invert. Originally, y of the wrist is a number between 0 and 1.0, where 0 is the top of the screen
            self.y_wrist = landmarks[0].y

            self.pointer_coord = (landmarks[8].x, landmarks[8].y)
            
            
            x1 = int(min(self.x_) * W) - 30
            y1 = int(min(self.y_) * H) - 30

            x2 = int(max(self.x_) * W) + 30
            y2 = int(max(self.y_) * H) + 30

            probs = self.model.predict_proba([np.asarray(self.data_aux)])
            self.confidence = np.max(probs)
            self.prediction = np.argmax(probs)
            self.predicted_gesture = self.labels_dict[int(self.prediction)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 4)
            cv2.putText(frame, f'{self.predicted_gesture} ({self.confidence*100:.2f}%)', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3, cv2.LINE_AA)

        return {"frame": self.frame, "predicted gesture": self.predicted_gesture, "confidence": self.confidence, "y_wrist": self.y_wrist, "pointer_coord": self.pointer_coord}