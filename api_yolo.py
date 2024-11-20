import torch
from torchvision import transforms
import time
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse, JSONResponse
import shutil
from pathlib import Path
import os
from typing import List
import cv2
import numpy as np
import pickle
from utilities import csv_converter, pose_to_num, get_pose_from_num, most_frequent, keypoints_parser, get_coords_line
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import tempfile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import requests

import supervision as sv
from pose_estimator_yolo import load_models, run_inference, draw_keypoints, process_video_frames

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    load_models()

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
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

input_url = ""
@app.post("/set_input_url/")
async def set_input_url(url: str):
    global input_url
    input_url = url
    return {
        "status": "success",
        "message": "Input URL set successfully",
        "url": input_url
    }
    
@app.get("/get_input_url/")
async def get_input_url():
    return {
        "url": input_url
    }

@app.get("/pose_video/{filename}")
async def pose_video(filename: str):
    video_path = UPLOAD_DIR / filename
    
    if not video_path.exists():
        return JSONResponse(content={
            "status": "error",
            "message": "Video file not found"
        }, status_code=404)

    return StreamingResponse(
        process_video_frames(str(video_path)),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/pose_video_url/")
async def pose_video_url():
    if not input_url:
        return JSONResponse(content={
            "status": "error",
            "message": "No input URL set. Please set input URL first."
        }, status_code=400)

    return StreamingResponse(
        process_video_frames(input_url),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    uvicorn.run("api_yolo:app", host="0.0.0.0", port=8080, reload=True)