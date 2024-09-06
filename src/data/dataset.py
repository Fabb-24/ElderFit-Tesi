import os
import numpy as np
import tqdm
import multiprocessing
from imblearn.over_sampling import SMOTE
from data.video import Video
from data.frame_mediapipe import Frame
import util
from dotenv import load_dotenv

load_dotenv()

class Dataset:

    def __init__(self):
        self.videos = []
        self.labels = []

    def oversampling(self, X, y):
        """
        Funzione che effettua l'oversampling tramite SMOTE delle classi minoritarie.

        Args:
        - X (numpy.ndarray): Le features del dataset.
        - y (numpy.ndarray): Le labels del dataset.

        Returns:
        - X_resampled (numpy.ndarray): Le features del dataset con oversampling.
        - y_resampled (numpy.ndarray): Le labels del dataset con oversampling.
        """

        X_shape = X.shape

        # Riformatta X per SMOTE (appiana le dimensioni)
        X = X.reshape(X_shape[0], -1)

        # Effettua l'oversampling
        smote = SMOTE(sampling_strategy='auto')
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Riformatta X in modo che abbia la forma originale
        X_resampled = X_resampled.reshape(-1, *X_shape[1:])

        return X_resampled, y_resampled

    @staticmethod
    def process_video(video_info):
        video = Video(video_info[0], video_info[1])
        #self.videos.append(video)
        print(f"{video_info[0]} processed")
        return video

    def create(self):
        """
        Funzione che crea il dataset per l'addestramento.
        """

        # Se il dataset esiste già lo elimino
        if os.path.exists(util.getDatasetPath()):
            for file in os.listdir(util.getDatasetPath()):
                os.remove(os.path.join(util.getDatasetPath(), file))

        # Ottengo le categorie di esercizi
        categories = [d for d in os.listdir(util.getVideoPath())]
        # salvo le categorie in un file npy
        np.save(os.path.join(util.getDatasetPath(), "categories.npy"), categories)
        labels = []
        parameters = {}
        
        for category in categories:
            video_category_path = os.path.join(util.getVideoPath(), category)
            subvideos = [d for d in os.listdir(video_category_path)]
            parameters[category] = []
            for video_name in tqdm.tqdm(subvideos, desc=f"Analizzo la categoria {category}", unit="video"):
                video_path = os.path.join(video_category_path, video_name)
                video = Video(video_path, category)
                parameters[category].append(video.get_parameters())
                keypoints = []
                opticalflow = []
                angles = []
                for window in video.get_windows():
                    keypoints.append(window.get_keypoints())
                    opticalflow.append(window.get_opticalflow())
                    angles.append(window.get_angles())
                util.save_data(keypoints, os.path.join(util.getDatasetPath(), "keypoints.npy"))
                util.save_data(opticalflow, os.path.join(util.getDatasetPath(), "opticalflow.npy"))
                util.save_data(angles, os.path.join(util.getDatasetPath(), "angles.npy"))
                labels.extend([category for _ in range(video.get_num_windows())])
                del video
        np.save(os.path.join(util.getDatasetPath(), "labels.npy"), labels)
        
        # Oversampling del dataset

        # Ottengo i dataset appena creati
        kp = np.load(os.path.join(util.getDatasetPath(), "keypoints.npy"), allow_pickle=True)
        of = np.load(os.path.join(util.getDatasetPath(), "opticalflow.npy"), allow_pickle=True)
        an = np.load(os.path.join(util.getDatasetPath(), "angles.npy"), allow_pickle=True)
        labels = np.load(os.path.join(util.getDatasetPath(), "labels.npy"), allow_pickle=True)
        # Concateno le features
        features = util.concatenate_features(kp, of)
        features = util.concatenate_features(features, an)
        # Applico l'oversampling
        features, labels = self.oversampling(features, labels)
        # Divido le features in keypoints e opticalflow e angles
        num_keypoints_data = Frame.num_keypoints_data
        num_opticalflow_data = Frame.num_opticalflow_data
        kp = features[:, :, :num_keypoints_data]
        of = features[:, :, num_keypoints_data:num_keypoints_data + num_opticalflow_data]
        an = features[:, :, num_keypoints_data + num_opticalflow_data:]
        # Salvo i nuovi dataset
        np.save(os.path.join(util.getDatasetPath(), "keypoints.npy"), kp)
        np.save(os.path.join(util.getDatasetPath(), "opticalflow.npy"), of)
        np.save(os.path.join(util.getDatasetPath(), "angles.npy"), an)
        np.save(os.path.join(util.getDatasetPath(), "labels.npy"), labels)
                
        # Creo il file dei parametri
        np.save(os.path.join(util.getParametersPath(), "parameters.npy"), parameters)
