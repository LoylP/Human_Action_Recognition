import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import time

def predict_realtime(model_path, label_encoder_path, video_path):
    model = load_model(model_path)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, 
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    
    sequence = []
    predictions = []
    threshold = 0.5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            sequence.append(landmarks.flatten())

            if len(sequence) == 30:
                input_data = np.expand_dims(sequence, axis=0)
                pred = model.predict(input_data)[0]
                predicted_class_idx = np.argmax(pred)
                confidence = pred[predicted_class_idx]

                if confidence > threshold:
                    predicted_action = label_encoder.classes_[predicted_class_idx]
                    cv2.putText(frame, f'Action: {predicted_action}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f'Confidence: {confidence:.2%}', (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                sequence = sequence[1:]  

        cv2.imshow('Real-time Action Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

model_path = "model/action_recognition_model.keras"
label_encoder_path = "model/label_encoder.pkl"
video_path = "/mnt/d/action/falling_down/S001C003P002R001A043_rgb.avi"

predict_realtime(model_path, label_encoder_path, video_path)