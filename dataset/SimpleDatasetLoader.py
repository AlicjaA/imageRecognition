import numpy as np
import cv2
import os


# datasets_name/class/image.jpg
class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths, verbose=-1):
        print("[INFO] processing")
        data = []
        labels = []
        for (i, image_paths) in enumerate(image_paths):
            #try:
                #pass
                #print(image_paths)
                image = cv2.imread(image_paths)
                label = image_paths.split(os.path.sep)[-2]
                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                cv2.imwrite('../PreProcessor/PreImages/'+str(i)+'.jpg', image)
                cv2.waitKey(0)
                data.append(image)
                labels.append(label)
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

            #except Exception as identifier:
                #print(identifier)
        print("[INFO] processing finished")
        return np.array(data), np.array(labels)
