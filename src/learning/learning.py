import numpy as np
import tensorflow as tf
from keras import Sequential
import os
from keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Conv2D, MaxPooling2D, Flatten
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

class Learning:
    """
    Classe che si occupa dell'apprendimento attraverso diversi modelli.
    Ogni funzione si occupa di addestrare un modello diverso.
    """
        
    @staticmethod
    def LSTM(X, y, save_folder = None, epochs=100):
        """
        Funzione che addestra il modello LSTM.
        Il modello LSTM è composto da due layer LSTM e due layer Dropout, seguiti da un layer Dense.

        Args:
        - X (np.ndarray): Array di input.
        - y (np.ndarray): Array di output.

        Returns:
        - model: Il modello addestrato.
        """

        # stampo info su X e y
        print("\nInfo su X e y:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        # ottengo il numero di valori diversi in y
        num_classes = len(np.unique(y))
        print(f"\nNumero di classi: {num_classes}")

        #X = X.astype(np.float32) / 255.0

        # Definizione del modello LSTM
        model = Sequential()
        model.add(Input(shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Compilazione del modello
        model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

        # Addestramento del modello
        print("\nAddestramento del modello...")
        model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1)

        # Salva il modello
        if save_folder is not None:
            model.save(os.path.join(save_folder, "LSTM.h5"))

        return model

    
    @staticmethod
    def combo_LSTM(X1, X2, y, save_folder = None, epochs=100):
        """
        Funzione che addestra il modello combo LSTM.
        Il modello è composto da 2 modelli LSTM, uno per i keypoints e uno per l'optical flow, entrambi formati da 2 layer LSTM e 2 layer Dropout.
        I due modelli LSTM sono concatenati e passati ad un layer Dense e infine all'output layer.

        Args:
        - X1 (np.ndarray): Array di input per i keypoints.
        - X2 (np.ndarray): Array di input per l'optical flow.
        - y (np.ndarray): Array di output.

        Returns:
        - model: Il modello addestrato.
        """

        # Stampa info su X1, X2 e y
        print("\nInfo su X1, X2 e y:")
        print(f"X1 shape: {X1.shape}")
        print(f"X2 shape: {X2.shape}")
        print(f"y shape: {y.shape}")

        # ottengo il numero di valori diversi in y
        num_classes = len(np.unique(y))
        print(f"\nNumero di classi: {num_classes}")

        # Genera un array di interi. L'array contiene il 10% dei numeri da 0 a len(X1) (casuali e senza ripetizioni)
        indices = np.random.choice(len(X1), int(len(X1) * 0.1), replace=False)
        # Dividi il dataset in training e test set in base agli indici generati
        X1_train, X1_test = np.delete(X1, indices, axis=0), X1[indices]
        X2_train, X2_test = np.delete(X2, indices, axis=0), X2[indices]
        y_train, y_test = np.delete(y, indices, axis=0), y[indices]

        # Input layers
        input_keypoints = Input(shape=(X1.shape[1], X1.shape[2]))
        input_opticalflow = Input(shape=(X2.shape[1], X2.shape[2]))

        # LSTM per keypoints
        lstm_keypoints = LSTM(64, return_sequences=False)(input_keypoints)
        lstm_keypoints = Dropout(0.5)(lstm_keypoints)
        #lstm_keypoints = LSTM(32, return_sequences=False)(lstm_keypoints)
        #lstm_keypoints = Dropout(0.5)(lstm_keypoints)
        

        # LSTM per optical flow
        lstm_opticalflow = LSTM(64, return_sequences=False)(input_opticalflow)
        lstm_opticalflow = Dropout(0.5)(lstm_opticalflow)
        #lstm_opticalflow = LSTM(32, return_sequences=False)(lstm_opticalflow)
        #lstm_opticalflow = Dropout(0.5)(lstm_opticalflow)

        # Concatena i due LSTM
        concatenated = Concatenate()([lstm_keypoints, lstm_opticalflow])

        # Dense layer
        dense = Dense(10, activation='relu')(concatenated)

        # Output layer
        output = Dense(num_classes, activation='softmax')(dense)

        # Creazione del modello
        model = Model(inputs=[input_keypoints, input_opticalflow], outputs=output)

        # Compila il modello
        model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

        # Addestra il modello
        print("\nAddestramento del modello...")
        model.fit([X1_train, X2_train], y, epochs=epochs, batch_size=32, validation_split=0.1)

        # Valuta il modello
        print("\nValutazione del modello...")
        loss, accuracy = model.evaluate([X1_test, X2_test], y_test)
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")

        # Salva il modello
        if save_folder is not None:
            model.save(os.path.join(save_folder, "combo_LSTM.h5"))
            
        return model
    

    @staticmethod
    def combo_3_LSTM(X1, X2, X3, y, save_folder = None, epochs=100):
        """
        Funzione che addestra il modello combo 3 LSTM.
        Il modello è composto da 3 modelli LSTM, uno per i keypoints, uno per l'optical flow e uno per gli angoli, tutti formati da 2 layer LSTM e 2 layer Dropout.
        I tre modelli LSTM sono concatenati e passati ad un layer Dense e infine all'output layer.

        Args:
        - X1 (np.ndarray): Array di input per i keypoints.
        - X2 (np.ndarray): Array di input per l'optical flow.
        - X3 (np.ndarray): Array di input per gli angoli.
        - y (np.ndarray): Array di output.

        Returns:
        - model: Il modello addestrato.
        """

        # Stampa info su X1, X2, X3 e y
        print("\nInfo su X1, X2, X3 e y:")
        print(f"X1 shape: {X1.shape}")
        print(f"X2 shape: {X2.shape}")
        print(f"X3 shape: {X3.shape}")
        print(f"y shape: {y.shape}")

        # ottengo il numero di valori diversi in y
        num_classes = len(np.unique(y))
        print(f"\nNumero di classi: {num_classes}")

        # Genera un array di interi. L'array contiene il 10% dei numeri da 0 a len(X1) (casuali e senza ripetizioni)
        indices = np.random.choice(len(X1), int(len(X1) * 0.1), replace=False)
        # Dividi il dataset in training e test set in base agli indici generati
        '''X1_train, X1_test = np.delete(X1, indices, axis=0), X1[indices]
        X2_train, X2_test = np.delete(X2, indices, axis=0), X2[indices]
        X3_train, X3_test = np.delete(X3, indices, axis=0), X3[indices]
        y_train, y_test = np.delete(y, indices, axis=0), y[indices]'''

        # Input layers
        input_keypoints = Input(shape=(X1.shape[1], X1.shape[2]))
        input_opticalflow = Input(shape=(X2.shape[1], X2.shape[2]))
        input_angles = Input(shape=(X3.shape[1], X3.shape[2]))

        # LSTM per keypoints
        lstm_keypoints = LSTM(64, return_sequences=True)(input_keypoints)
        lstm_keypoints = Dropout(0.5)(lstm_keypoints)
        lstm_keypoints = LSTM(32, return_sequences=False)(lstm_keypoints)
        lstm_keypoints = Dropout(0.5)(lstm_keypoints)
        
        # LSTM per optical flow
        lstm_opticalflow = LSTM(64, return_sequences=True)(input_opticalflow)
        lstm_opticalflow = Dropout(0.5)(lstm_opticalflow)
        lstm_opticalflow = LSTM(32, return_sequences=False)(lstm_opticalflow)
        lstm_opticalflow = Dropout(0.5)(lstm_opticalflow)

        # LSTM per angles
        lstm_angles = LSTM(64, return_sequences=True)(input_angles)
        lstm_angles = Dropout(0.5)(lstm_angles)
        lstm_angles = LSTM(32, return_sequences=False)(lstm_angles)
        lstm_angles = Dropout(0.5)(lstm_angles)

        # Concatena i tre LSTM
        concatenated = Concatenate()([lstm_keypoints, lstm_opticalflow, lstm_angles])

        # Dense layer
        dense = Dense(64, activation='relu')(concatenated)

        # Output layer
        output = Dense(num_classes, activation='softmax')(dense)

        # Creazione del modello
        model = Model(inputs=[input_keypoints, input_opticalflow, input_angles], outputs=output)

        # Compila il modello
        model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

        # Addestra il modello
        print("\nAddestramento del modello...")
        model.fit([X1, X2, X3], y, epochs=epochs, batch_size=32, validation_split=0.05)
        #model.fit([X1_train, X2_train, X3_train], y, epochs=epochs, batch_size=32, validation_split=0.05)

        # Valuta il modello
        '''print("\nValutazione del modello...")
        loss, accuracy = model.evaluate([X1_test, X2_test, X3_test], y_test)
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")'''

        # Salva il modello
        if save_folder is not None:
            model.save(os.path.join(save_folder, "combo_3_LSTM.h5"))

        return model
    

    @staticmethod
    def CNN(X, y, save_folder=None, epochs=100):
        """
        Funzione che addestra un modello CNN 2D per dati di input 15x68.
        """
        
        # stampo info su X e y
        print("\nInfo su X e y:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        # ottengo il numero di valori diversi in y
        num_classes = len(np.unique(y))
        print(f"\nNumero di classi: {num_classes}")

        # Normalizzazione dei dati (opzionale)
        X = X.astype(np.float32) / 255.0

        # Aggiungi dimensione dei canali (1 per immagini in scala di grigi)
        if X.ndim == 3:
            X = np.expand_dims(X, axis=-1)

        # Definizione del modello CNN
        model = Sequential()
        model.add(Input(shape=(X.shape[1], X.shape[2], X.shape[3])))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Compilazione del modello
        model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

        # Addestramento del modello
        print("\nAddestramento del modello...")
        model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1)

        # Salva il modello
        if save_folder is not None:
            model.save(os.path.join(save_folder, "CNN.h5"))

        return model
    

    @staticmethod
    def LSTM_RNN(X, y, save_folder=None, epochs=100):
        """
        Funzione che addestra un modello LSTM per dati di input 15x68.
        """
        
        # stampo info su X e y
        print("\nInfo su X e y:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        # ottengo il numero di valori diversi in y
        num_classes = len(np.unique(y))
        print(f"\nNumero di classi: {num_classes}")

        # Normalizzazione dei dati (opzionale)
        X = X.astype(np.float32) / 255.0

        # Modifica la forma dell'input per LSTM
        # LSTM si aspetta input con forma (batch_size, timesteps, features)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=-1)  # Aggiungi dimensione dei canali

        # Definizione del modello LSTM
        model = Sequential()
        model.add(Input(shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(32))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        # Compilazione del modello
        model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

        # Addestramento del modello
        print("\nAddestramento del modello...")
        model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1)

        # Salva il modello
        if save_folder is not None:
            model.save(os.path.join(save_folder, "LSTM_RNN.h5"))

        return model