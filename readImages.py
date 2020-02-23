from os import listdir
from PIL import Image as PImage

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        img = img.resize( [int(0.5 * s) for s in img.size] )
        loadedImages.append(img)

    return loadedImages