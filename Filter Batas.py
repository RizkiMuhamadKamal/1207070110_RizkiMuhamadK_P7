#Import Library
import matplotlib.pyplot as plt
from skimage import data
from skimage.io import imread
from skimage.color import rgb2gray 
import numpy as np

#Load & Plot Input Image
citra1 = imread(fname="mobil.tif")  # Membaca citra 1 dari file "mobil.tif"
citra2 = imread(fname="boneka2.tif")  # Membaca citra 2 dari file "boneka2.tif"

print('Shape citra 1 : ', citra1.shape)  # Menampilkan bentuk citra 1 (jumlah baris, jumlah kolom, jumlah channel)
print('Shape citra 1 : ', citra2.shape)  # Menampilkan bentuk citra 2 (jumlah baris, jumlah kolom, jumlah channel)

fig, axes = plt.subplots(1, 2, figsize=(10, 10))  # Membuat subplot dengan 1 baris dan 2 kolom untuk menampilkan citra input
ax = axes.ravel()  # Melakukan unravel pada objek axes

ax[0].imshow(citra1, cmap = 'gray')  # Menampilkan citra 1 sebagai citra input
ax[0].set_title("Citra 1")  # Memberikan judul pada citra 1

ax[1].imshow(citra2, cmap = 'gray')  # Menampilkan citra 2 sebagai citra input
ax[1].set_title("Citra 2")  # Memberikan judul pada citra 2

#Menyiapkan variable output
copyCitra1 = citra1.copy()  # Mengcopy citra 1 ke variabel copyCitra1
copyCitra2 = citra2.copy()  # Mengcopy citra 2 ke variabel copyCitra2

m1,n1 = copyCitra1.shape  # Mendapatkan jumlah baris dan kolom citra 1
output1 = np.empty([m1, n1])  # Membuat array kosong dengan ukuran m1 x n1 sebagai output citra 1

m2,n2 = copyCitra2.shape  # Mendapatkan jumlah baris dan kolom citra 2
output2 = np.empty([m2, n2])  # Membuat array kosong dengan ukuran m2 x n2 sebagai output citra 2
print('Shape copy citra 1 : ', copyCitra1.shape)  # Menampilkan bentuk copy citra 1
print('Shape output citra 1 : ', output1.shape)  # Menampilkan bentuk output citra 1

print('m1 : ',m1)  # Menampilkan nilai m1 (jumlah baris citra 1)
print('n1 : ',n1)  # Menampilkan nilai n1 (jumlah kolom citra 1)
print()

print('Shape copy citra 2 : ', copyCitra2.shape)  # Menampilkan bentuk copy citra 2
print('Shape output citra 3 : ', output2.shape)  # Menampilkan bentuk output citra 2
print('m2 : ',m2)  # Menampilkan nilai m2 (jumlah baris citra 2)
print('n2 : ',n2)  # Menampilkan nilai n2 (jumlah kolom citra 2)
print()

#Proses Filter Batas Pada Citra Input 1
for baris in range(0, m1-1):  # Melakukan perulangan sebanyak jumlah baris citra 1
    for kolom in range(0, n1-1):  # Melakukan perulangan sebanyak jumlah kolom citra 1
        
        a1 = baris  # Mengassign nilai baris ke variabel a1
        b1 = kolom  # Mengassign nilai kolom ke variabel b1
        
        arr = np.array([copyCitra1[a1-1, b1-1], copyCitra1[a1-1, b1], copyCitra1[a1, b1+1], \
            copyCitra1[a1, b1-1], copyCitra1[a1, b1+1], copyCitra1[a1+1, b1-1],  \
            copyCitra1[a1+1, b1], copyCitra1[a1+1, b1+1]])  # Membuat array dengan nilai piksel sekitar citra 1
        
        minPiksel = np.amin(arr)  # Menemukan nilai piksel minimum dari array
        maksPiksel = np.amax(arr)  # Menemukan nilai piksel maksimum dari array    
            
        if copyCitra1[baris, kolom] < minPiksel:  # Jika piksel citra 1 lebih kecil dari piksel minimum
            output1[baris, kolom] = minPiksel  # Assign piksel minimum ke output citra 1
        else:
            if copyCitra1[baris, kolom] > maksPiksel:  # Jika piksel citra 1 lebih besar dari piksel maksimum
                output1[baris, kolom] = maksPiksel  # Assign piksel maksimum ke output citra 1
            else:
                output1[baris, kolom] = copyCitra1[baris, kolom]  # Assign piksel citra 1 ke output citra 1

#Proses Filter Batas Pada Citra Input 2
for baris1 in range(0, m2-1):  # Melakukan perulangan sebanyak jumlah baris citra 2
    for kolom1 in range(0, n2-1):  # Melakukan perulangan sebanyak jumlah kolom citra 2
        
        a1 = baris1  # Mengassign nilai baris ke variabel a1
        b1 = kolom1  # Mengassign nilai kolom ke variabel b1
        
        arr = np.array([copyCitra2[a1-1, b1-1], copyCitra2[a1-1, b1], copyCitra2[a1, b1+1], \
            copyCitra2[a1, b1-1], copyCitra2[a1, b1+1], copyCitra2[a1+1, b1-1],  \
            copyCitra2[a1+1, b1], copyCitra2[a1+1, b1+1]])  # Membuat array dengan nilai piksel sekitar citra 2
        
        minPiksel = np.amin(arr)  # Menemukan nilai piksel minimum dari array
        maksPiksel = np.amax(arr)  # Menemukan nilai piksel maksimum dari array    
            
        if copyCitra2[baris1, kolom1] < minPiksel:  # Jika piksel citra 2 lebih kecil dari piksel minimum
            output2[baris1, kolom1] = minPiksel  # Assign piksel minimum ke output citra 2
        else:
            if copyCitra2[baris1, kolom1] > maksPiksel:  # Jika piksel citra 2 lebih besar dari piksel maksimum
                output2[baris1, kolom1] = maksPiksel  # Assign piksel maksimum ke output citra 2
            else:
                output2[baris1, kolom1] = copyCitra2[baris1, kolom1]  # Assign piksel citra 2 ke output citra 2

#Plot Citra Input dan Output Hasil dari Filter Batas
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Membuat subplot dengan 2 baris dan 2 kolom untuk menampilkan citra input dan output
ax = axes.ravel()  # Melakukan unravel pada objek axes

ax[0].imshow(citra1, cmap = 'gray')  # Menampilkan citra 1 sebagai citra input
ax[0].set_title("Input Citra 1")  # Memberikan judul pada citra 1

ax[1].imshow(citra2, cmap = 'gray')  # Menampilkan citra 2 sebagai citra input
ax[1].set_title("Input Citra 2")  # Memberikan judul pada citra 2

ax[2].imshow(output1, cmap = 'gray')  # Menampilkan output citra 1
ax[2].set_title("Output Citra 1")  # Memberikan judul pada output citra 1

ax[3].imshow(output2, cmap = 'gray')  # Menampilkan output citra 2
ax[3].set_title("Output Citra 2")  # Memberikan judul pada output citra 2

plt.show()  # Menampilkan plot citra input dan output