from skfuzzy import fuzzymath as fmath
import numpy as np
from PIL import ImageFilter

def convertToGray(imgs):
    loadedImages = []
    for img in imgs:
        img=img.convert('L')
        loadedImages.append(img)
        
    return loadedImages


def edge(imgs):
    loadedImages = []
    for img in imgs:
        #img=img.filter(ImageFilter.BLUR)
        img=img.filter(ImageFilter.SMOOTH_MORE)
        img=img.filter(ImageFilter.DETAIL)
        img=img.filter(ImageFilter.SHARPEN)
        img=img.filter(ImageFilter.FIND_EDGES)
        #img=img.filter(ImageFilter.CONTOUR)
        #img=img.filter(ImageFilter.EMBOSS)
        loadedImages.append(img)
        
    return loadedImages


def binarisation(imgs):
    loadedImages = []
    for img in imgs:
        #img=img.convert('L').point(lambda x: 0 if x<128 else 255, '1')
        img=img.convert('L').point(lambda x: 255 if x<30 else 0, '1')
        loadedImages.append(img)
        
    return loadedImages


