import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt

def get_dominant_colors(image, k=3):
  pixels = image.reshape((-1, 3))
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(pixels)
  centers = kmeans.cluster_centers_
  labels = kmeans.labels_
  counts = Counter(labels)
  sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
  dominant_colors = [centers[i[0]].astype(int) for i in sorted_counts]
  return dominant_colors

image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
dominant_colors = get_dominant_colors(image, k=5)
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.pie([1]*len(dominant_colors), colors=np.array(dominant_colors)/255)
plt.title("Dominant Colors")
plt.axis("equal")
plt.show()
