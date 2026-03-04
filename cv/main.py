import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

M = 1024
N = 768
img = np.zeros((M, N), dtype=np.uint8)
#plt.imshow(img, cmap='gray')
#plt.show()
cv.imshow('image', img)
if cv.waitKey(0) & 0xFF == 27:
    cv.destroyAllWindows()