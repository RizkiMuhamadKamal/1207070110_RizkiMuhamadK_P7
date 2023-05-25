# 1. LOW-PASS FILTERING
# import library
import cv2
import numpy as np
from matplotlib import pyplot as plt

# bgr
img = cv2.imread('kursi.jpg')  # Membaca gambar dengan format BGR menggunakan OpenCV

# rgb
mahe = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Mengubah format gambar dari BGR menjadi RGB menggunakan OpenCV

# tampilkan gambar awal tanpa filter
cv2.imshow('Original Image', img)  # Menampilkan gambar asli menggunakan OpenCV
cv2.waitKey(0)  # Menunggu tombol keyboard ditekan

# membuat filter: matriks berukuran 5 x 5
kernel = np.ones((5,5),np.float32)/25  # Membuat kernel berukuran 5x5 dengan nilai 1 dan membaginya dengan 25

# lakukan filtering
mahe_filter = cv2.filter2D(img,-1,kernel)  # Melakukan filtering gambar menggunakan kernel yang telah dibuat

# tampilkan gambar hasil filtering
cv2.imshow('Filtered Image', mahe_filter)  # Menampilkan gambar hasil filtering menggunakan OpenCV
cv2.waitKey(0)  # Menunggu tombol keyboard ditekan
cv2.destroyAllWindows()  # Menutup jendela gambar menggunakan OpenCV
plt.show()  # Menampilkan plot gambar menggunakan pyplot

# perbesar ukuran hasil plotting jika diperlukan
plt.rcParams["figure.figsize"] = (15,15)  # Mengatur ukuran gambar pada plot menggunakan pyplot

# plot pertama, gambar asli
plt.subplot(121),plt.imshow(mahe),plt.title('Original')  # Membuat subplot pertama untuk menampilkan gambar asli dengan judul 'Original'
plt.xticks([]), plt.yticks([])  # Menghilangkan sumbu x dan y pada plot pertama

# kedua, hasil filter
plt.subplot(122),plt.imshow(mahe_filter),  # Membuat subplot kedua untuk menampilkan gambar hasil filter
plt.title('Averaging')  # Memberikan judul 'Averaging' pada plot kedua
plt.xticks([]), plt.yticks([])  # Menghilangkan sumbu x dan y pada plot kedua

# Plot!
plt.show()  # Menampilkan plot gambar menggunakan pyplot

#=========================================================================================================================
#2. HIGH-PASS FILTERING
#=========================================================================================================================
# sebenarnya kita tidak perlu melakukan filtering lagi. Cukup sekali saja 
# di bagian awal, selama notebook ini tetap terhubung
import cv2
import numpy as np
from matplotlib import pyplot as plt

# memanggil citra sebagai grayscale (argument 0)
img = cv2.imread('Rizki.jpg',0)

# menerapkan algoritma high-pass filtering: laplacian
laplacian = cv2.Laplacian(img,cv2.CV_64F)

# sobel dengan ukuran kernel 5
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) # CV_64F menunjukkan nilai bit dari citra yang dihasilkan serta tipe datanya (F = Float)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

# perbesar ukuran hasil plotting 
plt.rcParams["figure.figsize"] = (20,20)

# menampilkan hasil filter
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()


# memanggil citra sebagai grayscale (argument 0)
img = cv2.imread('kucing.jpeg',0)

# memanggil fungsi Canny Edges dengan argument (citra, nilai_min, nilai_max)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

#Menggunakan OpenCV untuk Thresholding
img = cv2.imread('Rizki.jpg',0)

# Hitungan threshold. 
# Perhatikan nilai ambang batas bawah dan atas dari tiap fungsi 
# yang diberikan
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

# menampilkan hasil
titles = ['Gambar asli','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# menampilkan beberapa gambar sekaligus
for i in range(6):
    # 3 baris, 2 kolom
    plt.subplot(3,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


# masih menggunakan variabel img yang sama
# digunakan median blur untuk menghaluskan tepi objek pada citra
# ini diperlukan agar thresholding memberikan hasil lebih baik
img = cv2.medianBlur(img,5)

# Lakukan Thresholding
# Binary Threshold
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Adaptive Threshold dengan Mean
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

# Adaptive Threshold dengan Gaussian
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

# Plotting
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

# menampilkan hasil
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()