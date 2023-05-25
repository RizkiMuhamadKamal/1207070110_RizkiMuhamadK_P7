# Memanggil library yang dibutuhkan
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as inline
from skimage import data

# Mendefinisikan path direktori gambar
image_path = "kursi.jpg"

# Membaca gambar menggunakan OpenCV
img = cv2.imread(image_path)

# Mendapatkan dimensi gambar
row,column,_ = img.shape

# Membuat citra output dengan ukuran yang sama
img1 = np.zeros((row, column), dtype='uint8')

# Rentang nilai piksel yang digunakan sebagai batas
min_range = 10
max_range = 60

# Melakukan thresholding pada setiap piksel gambar
for i in range(row):
    for j in range(column):
        if img[i, j, 0] > min_range and img[i, j, 0] < max_range:
            img1[i, j] = 255
        else:
            img1[i, j] = 0

# Menampilkan citra input dan citra output beserta histogramnya
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title("Citra Input") # Menampilkan gambar yang diinput
ax[1].hist(img.ravel(), bins=256)
ax[1].set_title('Histogram Input') # Menampilkan Histogram dari gambar yang diinput
ax[2].imshow(img1, cmap=plt.cm.gray)
ax[2].set_title("Citra Output") # Menampilkan output
ax[3].hist(img1.ravel(), bins=256)
ax[3].set_title('Histogram Output') # Histogram dari Output

plt.show()