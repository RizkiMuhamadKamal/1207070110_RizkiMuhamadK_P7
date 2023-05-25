#Import Library
import matplotlib.pyplot as plt
from skimage import data
from skimage.io import imread
from skimage.color import rgb2gray 
import numpy as np
import cv2

#Load Image
citra1 = imread("kursi.jpg")  # Membaca gambar dengan nama file "mahe.jpg" dan menyimpannya dalam variabel citra1
print(citra1.shape)  # Menampilkan dimensi gambar citra1
plt.imshow(citra1, cmap='gray')  # Menampilkan gambar citra1 dengan peta warna abu-abu

#Proses Konvolusi
kernel = np.array([[-1, 0, -1],  # Membuat kernel konvolusi dengan matriks 3x3
                   [0, 4, 0], 
                   [-1, 0, -1]])

citraOutput = cv2.filter2D(citra1, -1, kernel)  # Melakukan konvolusi citra1 dengan kernel yang telah dibuat

fig, axes = plt.subplots(1, 2, figsize=(12, 12))  # Membuat subplot dengan 1 baris dan 2 kolom
ax = axes.ravel()  # Melakukan unravel pada objek axes

ax[0].imshow(citra1, cmap = 'gray')  # Menampilkan citra input (citra1)
ax[0].set_title("Citra Input")  # Memberikan judul pada citra input

ax[1].imshow(citraOutput, cmap = 'gray')  # Menampilkan citra output setelah proses konvolusi
ax[1].set_title("Citra Output")  # Memberikan judul pada citra output

plt.show()  # Menampilkan plot citra input dan output