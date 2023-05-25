# Import Library
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as inline
import cv2
import matplotlib.image as mpimg
from skimage import data

# Memanggil gambar
image_path = "kursi.jpg"

# Mengonversi gambar ke skala keabuan
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Melakukan Histogram Equalization
img_equalized = cv2.equalizeHist(image)

# Membuat objek CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Menerapkan CLAHE pada gambar asli
img_clahe = clahe.apply(image)

# Create an empty array to store the final output
image_cs = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')

# Melakukan Min-Max Contrasting
min_val = np.min(image)
max_val = np.max(image)

# Apply Min-Max Contrasting
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        image_cs[i, j] = 250.0 * (image[i, j] - min_val) / (max_val - min_val)


# Mengcopy gambar sebagai float untuk operasi berikutnya
copyimg = image.copy().astype(float)

m1, n1 = copyimg.shape
output1 = np.empty([m1, n1])

# Melakukan operasi perkalian konstanta pada setiap piksel gambar
for baris in range(0, m1 - 1):
    for kolom in range(0, n1 - 1):
        a1 = baris
        b1 = kolom
        output1[a1, b1] = copyimg[baris, kolom] * 1.9

# Menampilkan citra input dan citra output serta histogramnya
fig, axes = plt.subplots(5, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title("Citra Input")
ax[1].hist(image.ravel(), bins=256)
ax[1].set_title('Histogram Input')

ax[2].imshow(img_equalized, cmap=plt.cm.gray)
ax[2].set_title("Citra Output HE")
ax[3].hist(img_equalized.ravel(), bins=256)
ax[3].set_title('Histogram Output HE Method')

ax[4].imshow(image_cs, cmap=plt.cm.gray)
ax[4].set_title("Citra Output CS")
ax[5].hist(image_cs.ravel(), bins=256)
ax[5].set_title('Histogram Output CS Method')

ax[6].imshow(img_clahe, cmap=plt.cm.gray)
ax[6].set_title("Citra Grayscale CLAHE")
ax[7].hist(img_clahe.ravel(), bins=256)
ax[7].set_title('Histogram Output CLAHE Method')

ax[8].imshow(output1, cmap=plt.cm.gray)
ax[8].set_title("Citra Grayscale Perkalian Konstanta")
ax[9].hist(output1.ravel(), bins=256)
ax[9].set_title('Histogram Output Perkalian Konstanta Method')

fig.tight_layout()

plt.show()