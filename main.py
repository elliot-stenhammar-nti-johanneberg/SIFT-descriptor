#py -m pip install [packagename]

from PIL import Image, ImageFilter, ImageChops
import sys 
import numpy as np
import cv2

np.set_printoptions(threshold=sys.maxsize)

def importImage(image):
    img = Image.open(image)
    return img

def greyscaleImage(img):
    img_grey = img.convert('L')
    return img_grey

def resizeImage(img, width, height):
    resized_img = img.resize((width, height), Image.ANTIALIAS)
    return resized_img

# def GaussianFilter(img, sigma):
#     return img.filter(ImageFilter.GaussianBlur(radius = sigma))

# def DoG(img1, img2):
#     # temporary
#     return ImageChops.subtract(img1, img2)

def cropImage(img, width, height):
    left = -width/3
    top = 0
    right = 0
    bottom = height/3
    for i in range(1, 10):
        if right < width:
            left += width/3
            right += width/3
        else:
            left = 0
            right = width/3
            bottom += height/3
            top += height/3
        imgCrop = img.crop((left, top, right, bottom))
        imgCrop.save(str(i) + '.jpg')

def imageToArray(img):
    image_sequence = img.getdata()
    image_array = np.array(image_sequence)  
    # low value = black, high value = white
    for i in range(len(image_array)):
        if image_array[i] < 123:
            image_array[i] = True
        else: 
            image_array[i] = False
    # 1 = black, 0 = white
    return image_array

def direction(image_array):
    for i in range(len(image_array)):
        if image_array[i] 


width = 240
height = 240

# img = importImage("monkey.png")
# img = greyscaleImage(img)
# img = resizeImage(img, 240, 240)
# img_array = imageToArray(img) 



# img1 = GaussianFilter(img, 1)
# img2 = GaussianFilter(img, 0)
# img = DoG(img1, img2)
# img.show()

print(img_array)