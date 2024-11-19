from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import tempfile
import os
from pathlib import Path
import shutil
from typing import Optional
import asyncio
from util import predict_realtime
import uvicorn

app = FastAPI(
    title="Action Recognition API",
    description="API for real-time human action recognition",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL_PATH = "config/action_recognition_model.keras"
LABEL_ENCODER_PATH = "config/label_encoder.pkl"

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global variables for models
model = None
label_encoder = None

def load_models():
    """Load the model and label encoder"""
    global model, label_encoder
    try:
        model = load_model(MODEL_PATH)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Initialize models when the application starts"""
    load_models()

async def process_video_stream(video_path: str):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(str(video_path))
    sequence = []
    threshold = 0.5

    try:
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
                    mp_pose.POSE_CONNECTIONS
                )

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

            # Encode frame to JPEG
            _, encoded_frame = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
            
            await asyncio.sleep(0.033)  # ~30 FPS

    finally:
        cap.release()

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Verify file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return JSONResponse(content={
                "status": "error",
                "message": "Invalid file type. Please upload a video file."
            }, status_code=400)

        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse(content={
            "filename": file.filename,
            "status": "success",
            "message": "Video uploaded successfully"
        })
    
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/predict/{filename}")
async def predict_video(filename: str):
    video_path = UPLOAD_DIR / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        return StreamingResponse(
            process_video_stream(video_path),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/list_videos")
async def list_videos():
    try:
        videos = [f.name for f in UPLOAD_DIR.glob("*") if f.suffix.lower() in ('.mp4', '.avi', '.mov')]
        return JSONResponse(content={
            "status": "success",
            "videos": videos
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.delete("/delete_video/{filename}")
async def delete_video(filename: str):
    try:
        video_path = UPLOAD_DIR / filename
        if not video_path.exists():
            return JSONResponse(content={
                "status": "error",
                "message": "File not found"
            }, status_code=404)

        os.remove(video_path)
        return JSONResponse(content={
            "status": "success",
            "message": f"Video {filename} deleted successfully"
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/predict/{filename}")
async def predict_video(filename: str):
    video_path = UPLOAD_DIR / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    try:
        # Return the video file as a response with correct media type
        return FileResponse(video_path, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)