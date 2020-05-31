from Utils.ImageTools import ImageToArrayPreprocessor
from PreProcessor.Preprocessor import SimplePreprocessor
from dataset.SimpleDatasetLoader import SimpleDatasetLoader
from imutils import paths
import pickle
import argparse

def dataPreprocess():
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True, help="halloo insert 	dataset")
	ap.add_argument("-df", "--dataFile", required=False, help="provide data serializing file name", default='data.pkl')
	ap.add_argument("-lf", "--labelsFile", required=False, help="provide label serializing file name",
					default='labels.pkl')
	size = 50
	args = vars(ap.parse_args())

	print("[INFO] loading Images")
	imagePaths = list(paths.list_images(args["dataset"]))
	imagePaths[:] = [x for x in imagePaths if '.ipynb_checkpoints' not in x]
	sp = SimplePreprocessor(size, size)
	iap = ImageToArrayPreprocessor()
	sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
	(data, labels) = sdl.load(imagePaths, verbose=-1)

	output_data = open(f'./PreProcessor/PreProcessedData/'+args["dataFile"], 'wb')
	output_labels = open(f'./PreProcessor/PreProcessedData/'+args["labelsFile"], 'wb')
	pickle.dump(data, output_data, -1)
	pickle.dump(labels, output_labels, -1)
	output_data.close()
	output_labels.close()

if __name__=="__main__":
	dataPreprocess()