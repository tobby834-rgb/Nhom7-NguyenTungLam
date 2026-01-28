import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

#M = 1024
#N = 768
#img = np.zeros((M, N), dtype=np.uint8)
#cv.line(img, (0,0),(M-1,N-1),255)
#plt.imshow(img, cmap='gray')
#plt.show()
#cv.imshow('image', img)
#if cv.waitKey(0) & 0xFF == 27:
#    cv.destroyAllWindows()

#Sử dụng CV vẽ một đường tròn trùng tâm với tâm của ảnh, có bán kính là 100, màu trắng,độ dày 2 pixel.
#cv.circle(img, (M//2, N//2), 300, 125, 10)  
#cv.imshow('image with circle', img)
#if cv.waitKey(0) & 0xFF == 27:
#    cv.destroyAllWindows()


#Vẽ một bàn cờ vua đen trắng có kích thước 8x8, mỗi ô có kích thước 100x100 pixel.
chessboard = np.zeros((800, 800, 3), dtype=np.uint8)
#chessboard [0:99, 0:99] = (128, 0, 128)
#chessboard [100:199, 100:199] = (128, 0, 128)

for i in range(8):
    for j in range(8):
        if (i + j) % 2 == 0:
            cv.rectangle(chessboard, (i*100, j*100), ((i+1)*100, (j+1)*100), (128, 0, 128), -1)
        else:
            cv.rectangle(chessboard, (i*100, j*100), ((i+1)*100, (j+1)*100), (0, 255, 0), -1)
cv.imshow('bancovua', chessboard)
if cv.waitKey(0) & 0xFF == 27:
    cv.destroyAllWindows()
