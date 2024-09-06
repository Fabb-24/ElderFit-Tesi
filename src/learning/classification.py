import tensorflow as tf
import cv2
import numpy as np
import util
import os

from data.frame_mediapipe import Frame
from data.window import Window
from data.dataAugmentation import Videos
from learning.repetitions import Repetitions

    
def classify(model_path):
    # Ottengo il modello per la classificazione
    model = tf.keras.models.load_model(model_path)

    # Inizializza le ripetizioni
    repetitions = Repetitions()

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

        frames.append(Frame(frame))

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
            predicted_exercise.append(categories[prediction] if predictions[0][prediction] > 0.5 else "None")

            if len(predicted_exercise) == 3:
                predicted_exercise = predicted_exercise[1:]

            # ottengo l'esercizio presente piu volte in predicted_exercise
            effective_exercise = max(set(predicted_exercise), key=predicted_exercise.count)

            # Azzero le ripetizioni se l'esercizio predetto cambia
            if effective_exercise != last_predicted_exercise:
                repetitions.reset()

            last_predicted_exercise = effective_exercise
            frames = frames[int(util.getWindowSize()/2):]

        # Aggiorno le ripetizioni
        repetitions.update(frames[-1])

        # Disegna il nome dell'esercizio sulla frame
        cv2.putText(frame, f'Exercise: {effective_exercise} ({repetitions.get_category_rep(effective_exercise) if effective_exercise is not "None" else 0})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Mostra il frame con i keypoints
        cv2.imshow('Mediapipe Pose Estimation', frame)
        
        # Esci premendo 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Rilascia la videocamera e chiudi tutte le finestre
    cap.release()
    cv2.destroyAllWindows()








    '''windows = Windows(window_size)

    # Inizializzo il frame precedente a None
    prev_frame = None
    # Inizializzo le finestre di keypoints e optical flow
    window_keypoints = []
    window_opticalflow = []
    # Ottengo le categorie di esercizi dalle cartelle presenti in video
    categories = Videos(os.path.join(util.getBasePath(), "video")).get_categories()
    # Imposto la predizione iniziale a 0 (esercizio 0)
    exercise_prediction = 0
    # Inizializzo prediction_history con 5 elementi uguali a 0 (esercizio 0)
    prediction_history = [0] * 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Esegui inferenza
        keypoints = Keypoints(frame)

        if len(window_keypoints) < window_size:
            window_keypoints.append(keypoints)

            if prev_frame is not None:
                flow = keypoints.extract_opticalflow(prev_frame)
                window_opticalflow.append(flow)
            else:
                window_opticalflow.append(np.zeros((24,)))

        if len(window_keypoints) == window_size:
            X1 = windows.interpolate_keypoints_window(window_keypoints)
            # Creazione dell'input optical flow per il modello
            X2 = np.array(window_opticalflow)
            X2 = X2.astype(np.float32)
            X2 = X2.reshape(1, window_size, -1)
            # Creazione dell'input angles per il modello
            X3 = windows.create_angles_windows([X1])
            X3 = X3.astype(np.float32)
            X3 = X3.reshape(1, window_size, -1)
            # Creazione dell'input keypoints per il modello
            X1 = windows.process_keypoints_window(X1)
            X1 = X1.astype(np.float32)
            X1 = X1.reshape(1, window_size, -1)
            # Eseguo la predizione
            predictions = model.predict([X1, X2, X3], verbose=0)
            exercise_prediction = np.argmax(predictions, axis=1)[0]

            # faccio uno shift verso sinistra di 1 eliminando il primo elemento e aggiungendo exercise_prediction a destra in prediction_history
            prediction_history = prediction_history[1:] + [exercise_prediction]
            # ottengo il valore che compare piu volte in prediction_history
            exercise_prediction = max(set(prediction_history), key=prediction_history.count)
            # Di window_keypoints e window_opticalflow mantengo solo gli utlimi elementi di numero (window_size / 2)
            window_keypoints = window_keypoints[-int(window_size / 2):]
            window_opticalflow = window_opticalflow[-int(window_size / 2):]
        
        # Disegna il nome dell'esercizio sulla frame
        cv2.putText(frame, f'Esercizio: {categories[exercise_prediction]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # Disegna i keypoints sulla frame
        for i in range(keypoints.get_length()):
            x, y = int(keypoints.get_keypoint(i)["x"] * frame.shape[1]), int(keypoints.get_keypoint(i)["y"] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        # Mostra il frame con i keypoints
        cv2.imshow('MoveNet Pose Estimation', frame)
        # Salvo il frame precedente
        prev_frame = frame

        # Esci premendo 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia la videocamera e chiudi tutte le finestre
    cap.release()
    cv2.destroyAllWindows()'''