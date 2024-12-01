# Human Action Recognition

This project focuses on classifying and recognizing human actions in video sequences and real-time camera feeds. By leveraging cutting-edge deep learning techniques and frameworks, the system efficiently detects and interprets human actions, providing an intuitive and interactive user experience.

![](/config/image/UI.png)
---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The **Human Action Recognition Project** combines feature extraction, model training, and application deployment to create a scalable solution for human action recognition. It supports both single-person and multi-person scenarios, ensuring robust detection across various contexts.

## Features
### 1. Feature Extraction:

- **Single-Person Action Detection**: Utilizes MediaPipe for extracting skeletal keypoints.
- **Multi-Person Action Detection**: Employs YOLO Pose v7 to detect and extract keypoints from multiple individuals in the frame.

### 2. Model Training:
- A hybrid **CNN-LSTM** model is trained on the extracted features, combining the spatial understanding of CNNs with the temporal sequence analysis of LSTMs.
- This ensures accurate recognition of dynamic and complex human actions.
### 3. Demo Application:
- **Frontend**: Built using Next.js, offering a user-friendly interface for video and real-time action recognition.
- **Backend**: Developed with FastAPI, connecting the trained model for inference and managing the feature extraction and prediction pipeline.

## Technologies Used
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Next JS](https://img.shields.io/badge/Next-black?style=for-the-badge&logo=next.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) 
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) 

#### Frameworks & Libraries:
- MediaPipe: For single-person pose detection and feature extraction.
- YOLO Pose v7: For multi-person action detection.
- CNN-LSTM: For training the recognition model.
- FastAPI: For backend API development.
- Next.js: For frontend development and deployment.

## Installation
#### You can run with docker: 
```bash
# 1. Clone the repository:
git clone https://github.com/LoylP/Human_Action_Recognition.git
cd Human_Action_Recognition
# 2. Run docker-compose:
docker-compose up --build  
# 3. You can access system with:
Frontend: http://localhost:3000.
Backend_Mediapipe(FastAPI): http://localhost:8080/docs.
Backend_Yolo(FastAPI): http://localhost:8000.

# To stop the service:
docker-compose down  
```

## Usage
#### Start the Application:
```bash
docker-compose up  
```
#### Access the Frontend:
- Open your browser and navigate to http://localhost:3000.
- This is the demo application where you can upload video sequences or stream live video for human action recognition.
#### Upload a Video:
- Use the upload functionality to analyze a pre-recorded video.
- The application will extract features using MediaPipe for single-person actions.

![](/config/image/UI.png)

- The application will extract features using YOLO Pose V7 for multi-person scenarios.

![](/config/image/Multi.png)

#### Streaming camera with link/path:

![](/config/image/camera.png)
