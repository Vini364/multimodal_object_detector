import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

img = cv2.imread("/Users/vineshuniyal/Desktop/multimodal_object_detector/src/tr1.jpg")
img2 = plt.imread("/Users/vineshuniyal/Desktop/multimodal_object_detector/src/tr1.jpg")
if img is None:
  print("Nothing there!")
  exit()
histo = pd.Series(img.flatten()).plot(kind = "hist", title = "This is the histogram", bins = 50)

