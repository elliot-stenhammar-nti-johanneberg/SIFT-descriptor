from PIL import Image

def importImage(image):
    img = Image.open(image)
    return img

def greyscaleImage(img):
    img_grey = img.convert('L')
    return img_grey

def resizeImage(img):
    width = 240
    height = 240
    resized_img = img.resize((width, height), Image.ANTIALIAS)
    return resized_img


left = 0
top = 0
right = 80
bottom = 80

img = resizeImage(importImage("pog.png"))

img1 = img.crop((left, top, right, bottom))

img1.show()