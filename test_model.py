from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import argparse
from imutils import paths
import cv2
import pickle
import random


def show(image):
    # Figure size in inches
    plt.figure(figsize=(15, 15))

    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')
    plt.show()


def ReadyToUseImage(im):
    image_blur = cv2.GaussianBlur(im, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    min_red = np.array([80, 60, 140])
    max_red = np.array([255, 255, 255])
    image_red1 = cv2.inRange(image_blur_hsv, min_red, max_red)
    big_contour, mask = find_biggest_contour(image_red1)
    moments = cv2.moments(mask)
    centre_of_mass = (
        int(moments['m10'] / moments['m00']),
        int(moments['m01'] / moments['m00'])
    )
    image_with_com = im.copy()
    cv2.circle(image_with_com, centre_of_mass, 10, (0, 255, 0), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(im, im, mask=mask)

    return (dst)


def find_biggest_contour(image):
    image = image.copy()
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="halloo insert dataset")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    ap.add_argument("-df", "--dataFile", required=False, help="provide data serializing file name", default='data.pkl')
    ap.add_argument("-lf", "--labelsFile", required=False, help="provide label serializing file name",
                    default='labels.pkl')
    args = vars(ap.parse_args())
    imagePaths = list(paths.list_images(args["dataset"]))
    imagePaths[:] = [x for x in imagePaths if '.ipynb_checkpoints' not in x]
    size = 50


    print("[INFO] sampling images ...")
    data= pickle.load(open(f'./PreProcessor/PreProcessedData/'+args["dataFile"], 'rb'))
    labels = pickle.load(open(f'./PreProcessor/PreProcessedData/'+args["labelsFile"], 'rb'))
    simple_labels = np.unique(labels)
    simple_labels = simple_labels.astype(str)
    print("Labels: ", simple_labels )

    for_shuffle = list(zip(data, labels))
    random.shuffle(for_shuffle)
    data, labels = zip(*for_shuffle)
    data = np.asarray(data)
    labels = np.asarray(labels)


    data = data.astype("float") / 255.0
    labels = labels.astype(str)


    print("[INFO] loading pre-trained network ...")
    model = load_model(args["model"])
    print("[INFO] predicting ...")
    preds = model.predict(data, batch_size=size).argmax(axis=1)
    print(preds)
    print("[INFO] Results:")
    for (i, imagePath) in enumerate(imagePaths):
        # load the example image, draw the prediction, and display it
        # to our screen
        print("Predicted= %s, Should be= %s" % ( simple_labels[preds[i]], labels[i]))
        image = cv2.imread(imagePath)
        image=ReadyToUseImage(image)
        cv2.putText(image, "Label: {}".format(simple_labels[preds[i]]),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite('./TestResult/'+str(i)+'.jpg', image)
        cv2.waitKey(0)

if __name__=="__main__":
	main()
