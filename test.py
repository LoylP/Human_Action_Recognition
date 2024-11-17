import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import mediapipe as mp
import pickle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def extract_skeleton_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Chuyển đổi BGR sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Phát hiện pose
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Lấy các điểm landmark (33 điểm, mỗi điểm có x,y,z)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            frames_landmarks.append(landmarks.flatten())
    
    cap.release()
    
    # Chuẩn hóa số frame về cùng một độ dài
    if len(frames_landmarks) > 0:
        frames_landmarks = np.array(frames_landmarks)
        # Lấy 30 frame đều nhau từ video
        target_frames = 30
        if len(frames_landmarks) >= target_frames:
            indices = np.linspace(0, len(frames_landmarks)-1, target_frames, dtype=int)
            frames_landmarks = frames_landmarks[indices]
        else:
            # Padding nếu video quá ngắn
            padding = np.zeros((target_frames - len(frames_landmarks), frames_landmarks.shape[1]))
            frames_landmarks = np.vstack((frames_landmarks, padding))
        
        return frames_landmarks
    return None

def load_saved_model(model_path, label_encoder_path):
    model = load_model(model_path)
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder

def predict_video(video_path, model, label_encoder, confidence_threshold=0.5):
    features = extract_skeleton_features(video_path)
    
    if features is None:
        return "Không thể trích xuất skeleton từ video", 0.0
    
    features = np.expand_dims(features, axis=0)
    
    predictions = model.predict(features)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    if confidence < confidence_threshold:
        return "Không chắc chắn", confidence
        
    predicted_class = label_encoder.classes_[predicted_class_idx]
    return predicted_class, confidence

def predict_multiple_videos(video_folder, model_path, label_encoder_path, confidence_threshold=0.5):
    model, label_encoder = load_saved_model(model_path, label_encoder_path)
    
    for video_name in os.listdir(video_folder):
        if video_name.endswith(('.mp4', '.avi')):
            video_path = os.path.join(video_folder, video_name)
            action, confidence = predict_video(video_path, model, label_encoder, confidence_threshold)
            
            print(f"Video: {video_name}")
            print(f"Hành động dự đoán: {action}")
            print(f"Độ tin cậy: {confidence:.2%}")
            print("-" * 50)

model_path = "model/action_recognition_model.keras"
label_encoder_path = "model/label_encoder.pkl"
video_path = "/mnt/d/action/running/S020C003P043R001A099_rgb.avi"

model, label_encoder = load_saved_model(model_path, label_encoder_path)
action, confidence = predict_video(video_path, model, label_encoder, 0.5)
            
print(f"Hành động dự đoán: {action}")
print(f"Độ tin cậy: {confidence:.2%}")
print("-" * 50)