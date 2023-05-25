#Import Library
import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage.io import imread
from skimage.color import rgb2gray 
import numpy as np

#Load & Plot Input Image
citra1 = imread(fname="mobil.tif") # Membaca citra 1 dari file menggunakan skimage
citra2 = imread(fname="boneka2.tif") # Membaca citra 2 dari file menggunakan skimage

print('Shape citra 1 : ', citra1.shape) # Menampilkan dimensi citra 1
print('Shape citra 2 : ', citra2.shape) # Menampilkan dimensi citra 2

fig, axes = plt.subplots(1, 2, figsize=(10, 10)) # Membuat subplot dengan 1 baris dan 2 kolom untuk menampilkan citra 1 dan citra 2
ax = axes.ravel() # Melakukan unravel pada objek axes

ax[0].imshow(citra1, cmap='gray') # Menampilkan citra 1
ax[0].set_title("Citra 1") # Memberikan judul pada citra 1
ax[1].imshow(citra2, cmap='gray') # Menampilkan citra 2
ax[1].set_title("Citra 2") # Memberikan judul pada citra 2

#Menyiapkan variable output
copyCitra1 = citra1.copy().astype(float) # Meng-copy citra 1 ke variabel copyCitra1 dan mengubah tipe datanya menjadi float
copyCitra2 = citra2.copy().astype(float) # Meng-copy citra 2 ke variabel copyCitra2 dan mengubah tipe datanya menjadi float

m1,n1 = copyCitra1.shape # Mendapatkan ukuran baris dan kolom citra 1
output1 = np.empty([m1, n1]) # Membuat array kosong dengan ukuran yang sama dengan citra 1 untuk menyimpan hasil akhir

m2,n2 = copyCitra2.shape # Mendapatkan ukuran baris dan kolom citra 2
output2 = np.empty([m2, n2]) # Membuat array kosong dengan ukuran yang sama dengan citra 2 untuk menyimpan hasil akhir

print('Shape copy citra 1 : ', copyCitra1.shape) # Menampilkan dimensi copyCitra1
print('Shape output citra 1 : ', output1.shape) # Menampilkan dimensi output1

print('m1 : ',m1) # Menampilkan nilai m1 (jumlah baris citra 1)
print('n1 : ',n1) # Menampilkan nilai n1 (jumlah kolom citra 1)
print()

print('Shape copy citra 2 : ', copyCitra2.shape) # Menampilkan dimensi copyCitra2
print('Shape output citra 3 : ', output2.shape) # Menampilkan dimensi output2
print('m2 : ',m2) # Menampilkan nilai m2 (jumlah baris citra 2)
print('n2 : ',n2) # Menampilkan nilai n2 (jumlah kolom citra 2)
print()

#Proses Filter Rerata Pada Citra Input 1
for baris in range(0, m1-1): # Melakukan perulangan sebanyak jumlah baris citra 1
    for kolom in range(0, n1-1): # Melakukan perulangan sebanyak jumlah kolom citra 1
        a1 = baris
        b1 = kolom
        jumlah = copyCitra1[a1-1, b1-1] + copyCitra1[a1-1, b1] + copyCitra1[a1-1, b1-1] + \
                 copyCitra1[a1, b1-1] + copyCitra1[a1, b1] + copyCitra1[a1, b1+1] + \
                 copyCitra1[a1+1, b1-1] + copyCitra1[a1+1, b1] + copyCitra1[a1+1, b1+1] # Menghitung jumlah piksel dalam jendela 3x3
        output1[a1, b1] = (1/9 * jumlah) # Menghitung nilai rerata dan memasukkan hasilnya ke dalam output1

#Proses Filter Rerata Pada Citra Input 2
for baris1 in range(0, m2-1): # Melakukan perulangan sebanyak jumlah baris citra 2
    for kolom1 in range(0, n2-1): # Melakukan perulangan sebanyak jumlah kolom citra 2
        a1 = baris1
        b1 = kolom1
        jumlah = copyCitra2[a1-1, b1-1] + copyCitra2[a1-1, b1] + copyCitra2[a1-1, b1-1] + \
                 copyCitra2[a1, b1-1] + copyCitra2[a1, b1] + copyCitra2[a1, b1+1] + \
                 copyCitra2[a1+1, b1-1] + copyCitra2[a1+1, b1] + copyCitra2[a1+1, b1+1] # Menghitung jumlah piksel dalam jendela 3x3
        output2[a1, b1] = (1/9 * jumlah) # Menghitung nilai rerata dan memasukkan hasilnya ke dalam output2

#Plot Citra Input dan Output Hasil dari Filter Rerata
fig, axes = plt.subplots(2, 2, figsize=(10, 10)) # Membuat subplot dengan 2 baris dan 2 kolom untuk menampilkan citra-citra
ax = axes.ravel() # Melakukan unravel pada objek axes

ax[0].imshow(citra1, cmap='gray') # Menampilkan citra 1
ax[0].set_title("Input Citra 1") # Memberikan judul pada citra 1

ax[1].imshow(citra2, cmap='gray') # Menampilkan citra 2
ax[1].set_title("Input Citra 1") # Memberikan judul pada citra 2

ax[2].imshow(output1, cmap='gray') # Menampilkan output 1 hasil filter rerata
ax[2].set_title("Output Citra 1") # Memberikan judul pada output 1

ax[3].imshow(output2, cmap='gray') # Menampilkan output 2 hasil filter rerata
ax[3].set_title("Output Citra 2") # Memberikan judul pada output 2

plt.show() # Menampilkan plot hasil