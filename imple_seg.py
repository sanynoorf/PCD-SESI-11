import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

image = imageio.imread('image.png')

if len(image.shape) == 3:
    grayscale_image = np.mean(image, axis=2)
else:
    grayscale_image = image

sobel_x = sobel(grayscale_image, axis=0)  
sobel_y = sobel(grayscale_image, axis=1)  

sobel_edges = np.hypot(sobel_x, sobel_y)

threshold = np.mean(sobel_edges) 

segmented_image = sobel_edges > threshold

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(grayscale_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Edge Detection (Sobel)")
plt.imshow(sobel_edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Segmented Image (Thresholding)")
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

plt.show()