import tensorflow as tf
import torch
import numpy as np
import util
import os

from data.frame_mediapipe import Frame
from learning.functions import Rep_Good


class Classification:
    """
    Classe che si occupa di classificare l'esercizio eseguito.
    """

    def __init__(self, model_path):
        """
        Costruttore della classe.

        Args:
        - model_path (String): percorso del modello da utilizzare per la classificazione
        """

        # Se model path è un file che finisce con .h5, allora carico un modello keras
        if model_path.endswith(".h5"):
            self.model = tf.keras.models.load_model(model_path)
            self.model_lib = "keras"
        else:
            self.model = util.get_pytorch_model(model_path)
            self.model_lib = "pytorch"

        # Inizializzo le variabili
        self.frames = []
        self.last_frame = None
        self.categories = np.load(os.path.join(util.getDatasetPath(), "categories.npy"))
        self.predicted_exercise = []
        self.effective_exercise = "None"
        self.last_predicted_exercise = "None"
        self.rep_good = Rep_Good()


    def same_frame(self, frame1, frame2, threshold=0.05):
        """
        Funzione che riceve in input 2 frame e restituisce True se i keypoints sono molto simili tra loro e False altrimenti.
        La somiglianza è gestita da un valore di soglia.

        Args:
        - frame1 (Frame): primo frame
        - frame2 (Frame): secondo frame
        - threshold (float): valore di soglia per la somiglianza

        Returns:
        - bool: True se i frame sono simili, False altrimenti
        """

        keypoints1 = frame1.get_keypoints()
        keypoints2 = frame2.get_keypoints()
        if len(keypoints1) != len(keypoints2):
            return False
        for i in range(len(keypoints1)):
            if abs(keypoints1[i]["x"] - keypoints2[i]["x"]) > threshold or abs(keypoints1[i]["y"] - keypoints2[i]["y"]) > threshold:
                return False
        return True
    

    def classify(self, frame):
        """
        Funzione che riceve in input un frame e restituisce l'esercizio eseguito, il numero di ripetizioni e la frase associata.

        Args:
        - frame (numpy.ndarray): frame da classificare

        Returns:
        - effective_exercise (String): esercizio eseguito
        - rep (int): numero di ripetizioni
        - phrase (String): frase associata all'esercizio
        """

        curr_frame = Frame(frame)
        landmarks_o = curr_frame.get_keypoints()
        landmarks = []
        for landmark in landmarks_o:  # Creo una copia dei landmarks per evitare che vengano modificati i landmarks originali
            landmarks.append({
                "x": landmark["x"],
                "y": landmark["y"]
            })

        if len(self.frames) == 0 or not self.same_frame(self.frames[-1], curr_frame, threshold=0.04):  # Se il frame è diverso dal precedente, lo aggiungo alla lista
            self.frames.append(curr_frame)

        if len(self.frames) == util.getWindowSize():  # Se la lista ha raggiunto la dimensione della finestra
            opticalflow = []
            for i in range(len(self.frames)):  # Per ogni frame nella lista calcolo i keypoints, gli angoli e l'optical flow
                self.frames[i].interpolate_keypoints(self.frames[i - 1] if i > 0 else None, self.frames[i + 1] if i < len(self.frames) - 1 else None)
                self.frames[i].extract_angles()
                if i > 0:
                    opticalflow.append(self.frames[i].extract_opticalflow(self.frames[i - 1]))
                else:
                    opticalflow.append(np.zeros((Frame.num_opticalflow_data,)))
                
            # Creo i 3 input per il modello
            kp = np.array([self.frames[i].process_keypoints() for i in range(util.getWindowSize())])
            an = np.array([self.frames[i].process_angles() for i in range(util.getWindowSize())])
            of = np.array(opticalflow)
            X1 = kp.reshape(1, util.getWindowSize(), -1)
            X2 = of.reshape(1, util.getWindowSize(), -1)
            X3 = an.reshape(1, util.getWindowSize(), -1)

            # Eseguo la predizione
            if self.model_lib == "keras":
                predictions = self.model.predict([X1, X2, X3], verbose=0)
            else:
                predictions = self.model(torch.tensor(X1, dtype=torch.float32), torch.tensor(X2, dtype=torch.float32), torch.tensor(X3, dtype=torch.float32))
                predictions = predictions.detach().numpy()
            print(predictions)

            # Aggiorno lo storico delle predizioni
            prediction = np.argmax(predictions, axis=1)[0]
            self.predicted_exercise.append(self.categories[prediction] if predictions[0][prediction] > 0.8 else "None")
            # Riduco la lunghezza dello storico a 3 e calcolo l'esercizio effettivo come quello presente piu volte
            if len(self.predicted_exercise) == 4:
                self.predicted_exercise = self.predicted_exercise[1:]

            print(self.predicted_exercise)

            if len(self.predicted_exercise) == 3:
                self.effective_exercise = max(set(self.predicted_exercise), key=self.predicted_exercise.count)
            else:
                self.effective_exercise = self.predicted_exercise[-1]
                
            if (self.predicted_exercise[-1] != self.effective_exercise or len(self.predicted_exercise) == 1) and self.predicted_exercise[-1] != "None":
                self.rep_good.reset_category_count(self.predicted_exercise[-1])

            self.last_predicted_exercise = self.effective_exercise
            self.frames = self.frames[int(util.getWindowSize()/2):]

        # Se tutti i keypoints sono nulli, lo storico viene resettato, l'esercizio viene impostato a None, le ripetizioni vengono azzerate e la finestra viene svuotata
        if all([landmark["x"] == 0 and landmark["y"] == 0 for landmark in landmarks]):
            self.predicted_exercise = []
            self.effective_exercise = "None"
            self.frames = []
            self.rep_good.reset()

        # Aggiorno le ripetizioni
        self.rep_good.update(curr_frame)

        return self.effective_exercise, self.rep_good.get_category_rep(self.effective_exercise) if self.effective_exercise != "None" else 0, self.rep_good.get_category_phrase(self.effective_exercise) if self.effective_exercise != "None" else "", landmarks



'''def classify(model_path, callback=None):
    # Ottengo il modello per la classificazione
    model = tf.keras.models.load_model(model_path)

    # Inizializza le ripetizioni
    rep_good = Rep_Good()

    # Inizializza la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    frames = []
    categories = np.load(os.path.join(util.getDatasetPath(), "categories.npy"))
    predicted_exercise = []
    effective_exercise = "None"
    last_predicted_exercise = "None"


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        curr_frame = Frame(frame)
        #if not same_frame(frames[-1], curr_frame) if frames else True:
        frames.append(curr_frame)

        if len(frames) == util.getWindowSize():
            opticalflow = []
            for i in range(len(frames)):
                frames[i].interpolate_keypoints(frames[i - 1] if i > 0 else None, frames[i + 1] if i < len(frames) - 1 else None)
                frames[i].extract_angles()
                if i > 0:
                    opticalflow.append(frames[i].extract_opticalflow(frames[i - 1]))
                else:
                    opticalflow.append(np.zeros((Frame.num_opticalflow_data,)))
                
            kp = np.array([frames[i].process_keypoints() for i in range(util.getWindowSize())])
            an = np.array([frames[i].process_angles() for i in range(util.getWindowSize())])
            of = np.array(opticalflow)

            # Creo i 3 input per il modello
            X1 = kp.reshape(1, util.getWindowSize(), -1)
            X2 = of.reshape(1, util.getWindowSize(), -1)
            X3 = an.reshape(1, util.getWindowSize(), -1)

            # Eseguo la predizione
            predictions = model.predict([X1, X2, X3], verbose=0)
            prediction = np.argmax(predictions, axis=1)[0]
            predicted_exercise.append(categories[prediction] if predictions[0][prediction] > 0.4 else "None")

            if len(predicted_exercise) == 3:
                predicted_exercise = predicted_exercise[1:]

            # ottengo l'esercizio presente piu volte in predicted_exercise
            effective_exercise = max(set(predicted_exercise), key=predicted_exercise.count)

            # Azzero le ripetizioni se l'esercizio predetto cambia
            if effective_exercise != last_predicted_exercise:
                rep_good.reset()

            last_predicted_exercise = effective_exercise
            frames = frames[int(util.getWindowSize()/2):]

        # Aggiorno le ripetizioni
        rep_good.update(curr_frame)

        # Callback
        if callback:
            callback(frame, effective_exercise, rep_good.get_category_rep(effective_exercise) if effective_exercise != "None" else 0)
        
        # Esci premendo 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Rilascia la videocamera e chiudi tutte le finestre
    cap.release()
    cv2.destroyAllWindows()'''