from data.videoParams import VideoParams
from data.dataset import Dataset
import numpy as np
import os
import util

'''# leggo un file npy e stampo il contenuto
keypoints = np.load(os.path.join(util.getDatasetPath(), "keypoints.npy"), allow_pickle=True)
opticalflow = np.load(os.path.join(util.getDatasetPath(), "opticalflow.npy"), allow_pickle=True)
angles = np.load(os.path.join(util.getDatasetPath(), "angles.npy"), allow_pickle=True)

print(keypoints.shape)
print(opticalflow.shape)
print(angles.shape)'''

a = (1, 2)
print(a[0])