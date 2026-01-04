from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from scipy.cluster.vq import whiten
from scipy.cluster.vq import vq
from scipy.cluster.vq import kmeans
import numpy as np
import webcolors
from css3color import CSS3_RGB
image1 = cv2.imread("/Users/vineshuniyal/Desktop/multimodal_object_detector/src/tr1.jpg")
red_list = []
green_list = []
blue_list = []

image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image_reshape = image.reshape(-1,3)
for lines in image:
  for pixel in lines:
    temp_red, temp_green, temp_blue = pixel[0], pixel[1], pixel[2]
    red_list.append(temp_red)
    blue_list.append(temp_blue)
    green_list.append(temp_green)


def k_means_process(img_array):
  red = img_array[:,0]
  green = img_array[:,1]
  blue = img_array[:,2]
  df = pd.DataFrame({'red':red, 'green': green, 'blue':blue})
  df['scaled_red'] = whiten(df['red'])
  df['scaled_blue'] = whiten(df['blue'])
  df['scaled_green'] = whiten(df['green'])

  cluster_centers, distortion = kmeans(df[['scaled_red', 'scaled_green', 'scaled_blue']], 2)

  r_std,g_std,b_std = df[['red','green','blue']].std()
  labels, _ = vq(df[['red', 'green', 'blue']], cluster_centers)
  counts = np.bincount(labels)
  labels_max = np.argmax(counts)
  max_cluster_center = cluster_centers[labels_max]

  scaled_r = max_cluster_center[0]*r_std
  scaled_g = max_cluster_center[1]*g_std
  scaled_b = max_cluster_center[2]*b_std
  dominant_color = (scaled_r, scaled_g, scaled_b)
  int_color = tuple(map(int, dominant_color))
  return int_color

def closest_color(rgb_color):
  min_colors = {}
  for key, values in CSS3_RGB.items():
    r_c, g_c, b_c = values[0], values[1], values[2]
    rd = (r_c- rgb_color[0])**2
    gd = (g_c- rgb_color[1])**2
    bd = (b_c- rgb_color[2])**2
    min_colors[(rd+gd+bd)] = key
  return min_colors[min(min_colors.keys())]

# print(color_name)

# dominant_color_array = np.zeros((100, 100, 3), dtype=np.uint8)
# dominant_color_array[:, :] = dominant_color  # fill the whole image with the color
# plt.imshow(dominant_color_array)
# plt.axis('off')
# plt.show()



# color = []
# r_std,g_std,b_std = df[['red','green','blue']].std()
# for cluster_center in cluster_centers:
#   scaled_r, scaled_g, scaled_b = cluster_center
#   color.append((scaled_r * r_std / 255, scaled_g * g_std / 255, scaled_b * b_std / 255))
# max_tuple = max(color, key=lambda item: item[1])


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(red_list,green_list,blue_list, c = 'r', marker = 'o')
# ax.set_xlabel("X axis")
# ax.set_ylabel("Y axis")
# ax.set_zlabel("Z axis")
# plt.show()