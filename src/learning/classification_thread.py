import tensorflow as tf
import cv2
import numpy as np
from data.dataAugmentation import Videos
import util
import os
import threading
import queue
import mediapipe as mp
import tensorflow_hub as hub

from data.keypoints import Keypoints
from data.windows import Windows


class Classification:

    def __init__(self, model_path, window_size=45):
        # Coda condivisa tra i thread
        self.frame_queue = queue.Queue()
        self.prediction_queue = queue.Queue()

        # Creazione dei thread
        self.classification_thread = threading.Thread(target=self.classify)
        self.video_thread = threading.Thread(target=self.video)

        # Evento per la terminazione del thread
        self.stop_event = threading.Event()

        # Ottengo il modello per la classificazione
        self.model = tf.keras.models.load_model(model_path)
        self.window_size = window_size
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.model_keypoints = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/3")
        self.model_keypoints = self.model_keypoints.signatures["serving_default"]

    
    def start(self):
        # Avvio del thread
        self.classification_thread.start()
        # Avvio del video
        self.video_thread.start()


    def video(self):
        # Inizializza la webcam
        cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture(os.path.join(util.getBasePath(), "video", "arms_up_horizontal", "arms_up_completo.mp4"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # Ottengo la lista delle categorie di esercizi
        categories = Videos(os.path.join(util.getBasePath(), "video")).get_categories()
        # Inizializzo variabili utili
        frame_sent = 0
        prediction = 1

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Finché la webcam è attiva il thread non si interrompe
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Inserisco il frame nella coda
                if frame_sent < self.window_size:
                    self.frame_queue.put(frame)
                    frame_sent += 1

                try:
                    p = self.prediction_queue.get_nowait()
                    prediction = p
                    frame_sent = 0
                except queue.Empty:
                    pass

                # decommentare per visualizzare i keypoints
                # Trovo i keypoints tramite mediapipe
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Disegno i keypoints sul frame con cv2
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                cv2.putText(frame, f'Exercise: {util.getExerciseCategories()[prediction] if prediction != -1 else "None"}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                cv2.imshow('Mediapipe Exercise Classification', frame)

                # Esci premendo 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        self.stop_event.set()
        cv2.destroyAllWindows()


    def classify(self):
        windows = Windows(self.window_size)
        # Inizializzo il frame precedente a None
        prev_frame = None
        # Inizializzo le finestre di keypoints e optical flow
        window_keypoints = []
        window_opticalflow = []
        # Imposto la predizione iniziale a 0 (esercizio 0)
        exercise_prediction = 0

        while True:
            if self.stop_event.is_set():
                break

            try:
                frame = self.frame_queue.get_nowait()
                #keypoints = Keypoints(frame, self.model_keypoints)
                keypoints = Keypoints(frame)

                if len(window_keypoints) < self.window_size:
                    window_keypoints.append(keypoints)

                    if prev_frame is not None:
                        flow = keypoints.extract_opticalflow(prev_frame)
                        window_opticalflow.append(flow)
                    else:
                        window_opticalflow.append(np.zeros((Windows.get_num_opticalflow_data(),)))

                if len(window_keypoints) == self.window_size:
                    X1 = windows.interpolate_keypoints_window(window_keypoints)
                    # Creazione dell'input optical flow per il modello
                    X2 = np.array(window_opticalflow)
                    X2 = X2.astype(np.float32)
                    X2 = X2.reshape(1, self.window_size, -1)
                    # Creazione dell'input angles per il modello
                    X3 = windows.create_angles_windows([X1])
                    X3 = X3.astype(np.float32)
                    X3 = X3.reshape(1, self.window_size, -1)
                    # Creazione dell'input keypoints per il modello
                    X1 = windows.process_keypoints_window(X1)
                    X1 = X1.astype(np.float32)
                    X1 = X1.reshape(1, self.window_size, -1)

                    # Eseguo la predizione
                    predictions = self.model.predict([X1, X2, X3], verbose=0)
                    print(predictions)
                    # Se tutti i valori sono sotto la soglia di 0.5, la predizione è -1
                    if np.all(predictions < 0.6):
                        exercise_prediction = -1
                    else:
                        exercise_prediction = np.argmax(predictions, axis=1)[0]
                    self.prediction_queue.put(exercise_prediction)

                    # Svuoto le finestre di keypoints e optical flow
                    window_keypoints = []
                    window_opticalflow = []

                prev_frame = frame
            except queue.Empty:
                pass