import cv2 as cv
import numpy as np
import urllib.request

def read_img_url(url):
    req = urllib.request.urlopen(url)   # open the URL
    img_rw = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(img_rw, cv.IMREAD_COLOR)  # decode as color image
    return img
def add_noise(img):
    mean = 0
    sigma = 50
    noisy = np.random.normal(mean, sigma, img.shape)
    new_img = np.clip(img + noisy, 0, 255).astype(np.uint8)
    return new_img
def add_muoi_tieu (img, prob=0.03):
    output = np.copy(img)
    row, col, ch = img.shape
    # Salt (white pixels)
    num_salt = int(prob * row * col)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    output[coords[0], coords[1]] = 255

    # Pepper (black pixels)
    num_pepper = int(prob * row * col)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    output[coords[0], coords[1]] = 0

    return output


if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_images/solidWhiteCurve.jpg"
    anh_goc = read_img_url(url)
    anh_muoi_tieu = add_muoi_tieu(anh_goc, 0.03)
    img2 = anh_muoi_tieu.copy()
    clean_img = cv.blur(img2, (3,3))
    img3 = anh_muoi_tieu.copy()
    clean_img = cv.medianBlur(img3, 5)
    img4 = np.concatenate((anh_goc, anh_muoi_tieu, clean_img), axis=1)
    cv.imshow("img4", img4)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ed1 = cv.Canny(anh_muoi_tieu, 50, 100)
    ed2 = cv.Canny(clean_img, 50, 100)
    ed3 = cv.Canny(anh_goc, 50, 100)
    img5 = np.concatenate((ed1, ed2, ed3), axis=1)
    cv.imshow("img5", img5)
    cv.waitKey(0)
    cv.destroyAllWindows() 
    cv.imshow("ed2", ed2)
    cv.waitKey(0)
    cv.destroyAllWindows()