from detector import model2_detect, model2, convert_to_tensor
from webcam import get_camera, read_camera
import cv2
import time
import datetime
import csv
from scipy.cluster.vq import whiten
import pandas as pd
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
import numpy as np
from clustercheck import k_means_process, closest_color
camera = get_camera() #Camera feed

if not camera.isOpened():
  print("Camera isnt opening")
  exit()

########VALUES##########
prev_frametime = 0
current_frametime = 0
frame_count = 0
confidence_t = [0.3, 0.5, 0.7]
class_filter = ['All', 'Person', 'Objects', 'Vehicles']
class_filter_index = 0
headers = ["timestamp", "class_name", "confidence", "xmin", "ymin", "xmax", "ymax"]
detection_time = 0
results = None
new_index = 0
annoted = None
detections_list = []
logging = False
user_command = None
command_active = False
###########################

######CREATION OF FILE##########
with open("hello.csv", mode = "w", newline= "") as file:
  writer = csv.writer(file)
  writer.writerow(headers)

while True:
  ret, frame = read_camera(camera)
  frame_count+=1
  if not ret:
    break

  if(frame_count%3 == 0): #FRAME SKIPPING TO REDUCE 
    start = time.time()
    results = model2_detect(frame)
    detection_time = time.time()-start 
    detections = results.xyxy[0]
    detections_list = []

    ######CLASS FILTER LOOP#######
    for i in detections: 
      class_index = i[5]
      if class_filter_index == 0:
        detections_list.append(i.tolist())
      elif(class_filter_index == 1 and class_index == 0):
        detections_list.append(i.tolist())
      elif(class_filter_index == 2 and class_index not in [0,1,2,3,5,6,7,8]):
        detections_list.append(i.tolist())
      elif(class_filter_index == 3 and class_index in [1,2,3,5,6,7,8]):
        detections_list.append(i.tolist())
    results.xyxy[0] = convert_to_tensor(detections_list)
    ######################################

  if results is not None:
    current_frametime = time.time()
    annoted = results.render()[0]
    annoted1 = annoted.copy()
    for i in detections:
      xmin, ymin, xmax, ymax = int(i[0].item()), int(i[1].item()), int(i[2].item()), int(i[3].item())
      roi = annoted1[ymin:ymax, xmin:xmax]
      if roi.size == 0:
        continue
      image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
      pixels = image_rgb.reshape(-1, 3)
      # most_color = k_means_process(pixels)
      median_color = np.median(image_rgb.reshape(-1, 3), axis=0)
      closest = closest_color(median_color)
      cv2.putText(annoted1, f"This is the closest color to the object: {closest}", (xmin, max(ymin-10,20)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2) 

    fps = 1/(current_frametime - prev_frametime)
    prev_frametime = current_frametime
    int_fps = int(fps)

    dictionary = {}
    for i in range(detections.shape[0]):
      class_id = int(detections[i,5])
      if class_id in dictionary:
        dictionary[class_id] +=1
      else:
        dictionary[class_id] = 1
    count = detections.shape[0]


    cv2.rectangle(annoted1, (0,0), (420, 200), (0,0,255), 2)
    for key, value in dictionary.items():
      line_height = 1 
      y = 120 + key*line_height
      cv2.putText(annoted1, f"Count for {model2.model.names[key]} is {value}", (20, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)

    cv2.putText(annoted1, f"Here is detection time {detection_time:3f}", (4, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2) #Detection Time
    cv2.putText(annoted1, f"Here is FPS time {int_fps:.3f}", (7,70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2) #FPS TIME
    cv2.putText(annoted1, f"Here is object count {count}", (13, 90), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2) #COUNTER
    cv2.putText(annoted1, f"Here is the confidence threshold: {model2.conf}", (20,200), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2) #CONFIDENCE THRESHOLD DISPLAY
    
    cv2.imshow("Camera", annoted1)

   #######LOGGING STATE LOOP#########
  if results is not None and logging:
    rows = []
    for i in detections:
      rows.append([time.time(), model2.model.names[int(i[5])], i[4].item(), i[0].item(), i[1].item(), i[2].item(), i[3].item()])
    ##################################

  ######KEY and STATES########
  k = cv2.waitKey(1) & 0xFF
  if k == 27:
    break
  elif k == ord('s'):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    str_current_datetime = str(current_datetime)
    file_name = str_current_datetime
    cv2.imwrite(f"{file_name}.png", annoted1)
    continue
  elif k == ord('c'):
    if(new_index>=len(confidence_t)):
      new_index = 0
    model2.conf = confidence_t[new_index]
    new_index +=1
  elif k == ord('0'):
      class_filter_index+=1
      if(class_filter_index>=len(class_filter)):
        class_filter_index = 0
  elif k == ord('l'):
    logging = not logging
    if logging:
        print("Logging state ON")
        continue   
    else:
      with open("hello.csv", mode = "a", newline = "") as file:
        writer = csv.writer(file)
        writer.writerows(rows)
      print("Logging state OFF")

camera.release()
cv2.destroyAllWindows()