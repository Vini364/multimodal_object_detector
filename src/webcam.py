import cv2

def get_camera():
  return cv2.VideoCapture(0)

def read_camera(camera):
  return camera.read()
