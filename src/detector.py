import torch
from ultralytics.solutions import ObjectCounter
import cv2
# Define the input image source (URL, local file, PIL image, OpenCV frame, numpy array, or list)
img = "/Users/vineshuniyal/Desktop/multimodal_object_detector/src/tr1.jpg"  # Example image

# model = torch.hub.load("ultralytics/yolov5", "yolov5s") 
model2 = torch.hub.load("ultralytics/yolov5", "yolov5n")

# def model_detect(image):
#   results = model(image)
#   return results

def model2_detect(image):
  results2 = model2(image)
  return results2

def convert_to_tensor(list):
  return torch.tensor(list, dtype = float)
