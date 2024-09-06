import os
from dotenv import load_dotenv
import numpy as np
import mediapipe as mp
import tensorflow as tf
import tensorflow_hub as hub

load_dotenv()


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")


def getExerciseCategories():
    """
    Funzione che restituisce la lista delle categorie di esercizi

    Returns:
        list: la lista delle categorie di esercizi
    """

    return [ct for ct in os.getenv("EXERCISE_CATEGORIES").split(",")]


def calculate_angle(a, b, c):
    """
    Funzione che, dati 3 punti con le loro coordinate x e y, restituisce l'ampiezza dell'angolo in gradi

    Args:
    - a (dict): primo angolo
    - b (dict): secondo angolo
    - c (dict): terzo angolo

    Returns:
    - angle (double): angolo in gradi
    """
    radians = np.arctan2(c["y"] - b["y"], c["x"] - b["x"]) - np.arctan2(a["y"] - b["y"], a["x"] - b["x"])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def calculate_distance(a, b):
    """
    Funzione che calcola la distanza euclidea tra due punti.

    Args:
    - p1 (Array): primo punto
    - p2 (Array): secondo punto

    Returns:
    - float: distanza euclidea tra i due punti
    """

    return np.sqrt((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2)


def save_data(data, file_path):
    """
    Funzione che salva dati nel file in modalità append

    Args:
    - data: dati da aggiungere al file
    - file_path (string): percorso del file in cui salvare i dati

    Returns:
    - (bool): vero se i dati sono stati salvati correttamente, falso altrimenti
    """

    try:
        if os.path.exists(file_path):
            existing_data = np.load(file_path, allow_pickle=True)
            data = np.concatenate((existing_data, data))
        np.save(file_path, data)
        return True
    except Exception as e:
        return False
    

def concatenate_features(features1, features2):
    """
    Funzione che concatena due insiemi di features.

    Args:
    - features1 (numpy.ndarray): Il primo insieme di features.
    - features2 (numpy.ndarray): Il secondo insieme di features.

    Returns:
    - concatenated_features (numpy.ndarray): Le features concatenate.
    """

    return np.concatenate([features1, features2], axis=2)


# ================================================================== #
# ================ FUNZIONI PER IL RECUPERO DEI PATH =============== #
# ================================================================== #

def getBasePath():
    """
    Metodo che ritorna il path base del progetto

    Returns:
        str: il path base del progetto
    """

    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "..")


def getVideoPath():
    """
    Funzione che restituisce il percorso della cartella dei video

    Returns:
        str: il percorso della cartella dei video
    """

    return os.path.join(getBasePath(), os.getenv("VIDEO_PATH"))


def getDatasetPath():
    """
    Funzione che restituisce il percorso della cartella del dataset

    Returns:
        str: il percorso della cartella del dataset
    """

    return os.path.join(getBasePath(), os.getenv("DATASET_PATH"))


def getModelsPath():
    """
    Funzione che restituisce il percorso della cartella dei modelli

    Returns:
        str: il percorso della cartella dei modelli
    """

    return os.path.join(getBasePath(), os.getenv("MODELS_PATH"))


def getParametersPath():
    """
    Funzione che restituisce il percorso della cartella dei parametri

    Returns:
        str: il percorso della cartella dei parametri
    """

    return os.path.join(getBasePath(), os.getenv("PARAMETERS_PATH"))


def getWindowSize():
    """
    Funzione che restituisce la dimensione della finestra

    Returns:
        int: la dimensione della finestra
    """

    return int(os.getenv("WINDOW_SIZE"))