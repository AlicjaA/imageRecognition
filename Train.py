from sklearn.preprocessing import LabelBinarizer
#from sklearn.preprocessing import OneHotEncoder
#import sys
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Utils.ImageTools import ImageToArrayPreprocessor
from PrePorcessor.Preprocessor import SimplePreprocessor
from dataset.SimpleDatasetLoader import SimpleDatasetLoader
from keras.utils import to_categorical
from Model.IncludeNet import IncludeNet
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
from imutils import paths
import random
from keras.callbacks import EarlyStopping

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True, help="halloo insert 	dataset")
	ap.add_argument("-m", "--model", required=True, help="path to output model")
	ap.add_argument("-e", "--epoches", required=True, help="how many epoches?")
	ap.add_argument("-l", "--learningRate", required=False, help="provide learning rate", default=0.025)
	ap.add_argument("-mn", "--momentum", required=False, help="provide momentum", default= 0.4)
	args = vars(ap.parse_args())
	size = 50
	ep = int(args["epoches"])
	dpt = 3
	classes = 4
	lr= float(args["learningRate"])
	momentum=float(args["momentum"])


	print("[INFO] loading Images")
	imagePaths = list(paths.list_images(args["dataset"]))
	#while '.ipynb_checkpoints' in imagePaths: imagePaths.remove('.ipynb_checkpoints')
	imagePaths[:] = [x for x in imagePaths if '.ipynb_checkpoints' not in x]
	sp = SimplePreprocessor(size, size)
	iap = ImageToArrayPreprocessor()
	sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
	(data, labels) = sdl.load(imagePaths, verbose=-1)
	#np.set_printoptions(threshold=sys.maxsize)

	for_shuffle = list(zip(data,labels ))
	random.shuffle(for_shuffle)
	data,labels = zip(*for_shuffle)
	data = np.asarray(data)
	labels = np.asarray(labels)


	print(list(dict.fromkeys(labels)))

	data = data.astype("float") / 255.0

	print("[INFO] splitting Images")
	#(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
	trainX = data[:int(len(data)*.8)]
	trainY = labels[:int(len(data)*.8)]
	testX = data[int(len(data)*.8):]
	testY = labels[int(len(data)*.8):]
	
	print("[INFO] LabelBinarizer")
	trainY = LabelBinarizer().fit_transform(trainY)
	testY = LabelBinarizer().fit_transform(testY)
	print('trainX: ', trainX.shape, ', trainY: ', trainY.shape, ', testX: ', testX.shape,', testY: ', testY.shape )
	
	#trainY = OneHotEncoder().fit_transform(trainY.reshape(-1, 1))
	#testY = OneHotEncoder().fit_transform(testY.reshape(-1, 1))
	#print('trainX: ', trainX.shape, ', trainY: ', trainY.shape, ', testX: ', testX.shape,', testY: ', testY.shape, )
	
	#trainY = to_categorical(trainY, classes)
	#testY = to_categorical(testY, classes)
	#print('trainX: ', trainX.shape, ', trainY: ', trainY.shape, ', testX: ', testX.shape,', testY: ', testY.shape, )
	
	print("[INFO] compiling model...")
	opt = SGD(lr=lr, momentum=momentum)
	model = IncludeNet.build(width=size, height=size, depth=dpt, classes=classes)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	
	print("[INFO] training network...")
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
	H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=size, epochs=ep, verbose=1, callbacks=[es])
	print("Saving network")
	model.save(f'./SavedModel/{args["model"]}')
	print("Network have been saved")
	
	print("[INFO] evaluating network...")
	predictions = model.predict(testX, batch_size=size)
	print(classification_report(testY.argmax(axis=1),
	                            predictions.argmax(axis=1),
	                            target_names=['EOSINOPHIL', 'NEUTROPHIL', 'MONOCYTE', 'LYMPHOCYTE']))
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, ep), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, ep), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, ep), H.history["acc"], label="accuracy")
	plt.plot(np.arange(0, ep), H.history["val_acc"], label="val_acc")
	plt.title("AMINJAMAL")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/ACC")
	plt.legend()
	plt.show()
	
if __name__=="__main__":
	main()
