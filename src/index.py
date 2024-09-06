from data.dataset import Dataset
from data.dataAugmentation import Videos
from learning.learning import Learning
#from learning.classification_thread import Classification
import learning.classification as Classification

import util
import os
import numpy as np


videos = Videos(os.path.join(util.getBasePath(), "video"))
#dataset = Dataset(os.path.join(util.getBasePath(), "video"), os.path.join(util.getBasePath(), "dataset"))


def process_videos():
    videos.process_videos()


def create_dataset():
    Dataset().create()
    

def create_model():
    keypoints = np.load(os.path.join(util.getDatasetPath(), "keypoints.npy"))
    opticalflow = np.load(os.path.join(util.getDatasetPath(), "opticalflow.npy"))
    angles = np.load(os.path.join(util.getDatasetPath(), "angles.npy"))
    labels = np.load(os.path.join(util.getDatasetPath(), "labels.npy"))
    categories = np.load(os.path.join(util.getDatasetPath(), "categories.npy"))
    # Creo un vettore uguale a labels ma dove al posto della stringa dell'esercizio ci metto l'indice corrispondente del vettore delle categorie
    categories = videos.get_categories()
    labels = np.array([categories.index(label) for label in labels])
    # Addestramento
    Learning.combo_3_LSTM(keypoints, opticalflow, angles, labels, os.path.join(util.getBasePath(), "models"), epochs=20)


def classify(window_size=15):
    Classification.classify(os.path.join(util.getBasePath(), "models", "combo_3_LSTM.h5"))
    #classification = Classification(os.path.join(util.getBasePath(), "models", "combo_3_LSTM.h5"), window_size=window_size)
    #classification.start()


if __name__ == "__main__":
    #process_videos()
    #create_dataset()
    #create_model()
    classify()