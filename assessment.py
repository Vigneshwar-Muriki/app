import subprocess

subprocess.run(['pip', 'install', 'torch', 'torchvision', 'opencv-python', 'zipfile36'], check=True)


import time
import requests

url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
dataset_tar = 'faces_dataset.tgz'

for attempt in range(5):  # Try 5 times
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(dataset_tar, 'wb') as f:
                f.write(r.content)
            print("Download successful")
            break
    except requests.exceptions.ConnectTimeout:
        print(f"Attempt {attempt+1}: Connection timed out, retrying in 5 seconds...")
        time.sleep(5)
    except Exception as e:
        print(f"Attempt {attempt+1}: An error occurred: {e}")
        time.sleep(5)

import cv2
import os

# Define the path to the dataset
dataset_path = 'faces_dataset/lfw-deepfunneled'

# Load the images and convert them to a format compatible with YOLO
image_paths = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".jpg"):
            image_paths.append(os.path.join(root, file))

# Display the first image to verify
img = cv2.imread(image_paths[0])
cv2.imshow("Image", img)
cv2.waitKey(0)

import torch
from torchvision import models

# Load a pre-trained YOLO model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the model for fine-tuning (e.g., updating the head layer for face detection)
num_classes = 2  # Background and Face
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

# Define the training loop
def train_model(model, dataset):
    # Training logic to fine-tune the model goes here
    pass

# Fine-tune the model on the new dataset
train_model(model, image_paths)

# Save the model
person_name = "vigneshwar_muriki"
model_path = f'Downloads/output/{person_name}_model.pth'
torch.save(model.state_dict(), model_path)