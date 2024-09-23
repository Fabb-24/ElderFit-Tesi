import numpy as np
import tensorflow as tf
from keras import Sequential
import os
from keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Conv2D, MaxPooling2D, Flatten
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
    def combo_3_LSTM(X1, X2, X3, y, X1_test, X2_test, X3_test, y_test, num_classes, save_folder = None, epochs=20):
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

        '''# ottengo il numero di valori diversi in y
        num_classes = len(np.unique(y))
        print(f"\nNumero di classi: {num_classes}")'''

        '''# Genera un array di interi. L'array contiene il 20% dei numeri da 0 a len(X1) (casuali e senza ripetizioni)
        indices = np.random.choice(len(X1), int(len(X1) * 0.2), replace=False)
        # Dividi il dataset in training e test set in base agli indici generati
        X1, X1_test = np.delete(X1, indices, axis=0), X1[indices]
        X2, X2_test = np.delete(X2, indices, axis=0), X2[indices]
        X3, X3_test = np.delete(X3, indices, axis=0), X3[indices]
        y, y_test = np.delete(y, indices, axis=0), y[indices]'''

        '''# Ottengo il dataset di test dalla cartella "dataset_test"
        X1_test = np.load("dataset_test/keypoints.npy")
        X2_test = np.load("dataset_test/opticalflow.npy")
        X3_test = np.load("dataset_test/angles.npy")
        y_test = np.load("dataset_test/labels.npy")
        cat = np.load("dataset_test/categories.npy")
        y_test = np.array([list(cat).index(label) for label in y_test])'''

        # trasformo ogni y da numero a array di 0 e 1 dove 1 è la posizione del numero
        #y = tf.keras.utils.to_categorical(y, num_classes)
        #y_test = tf.keras.utils.to_categorical(y_test, num_classes)


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
        output = Dense(num_classes + 1, activation='softmax')(dense)

        # Creazione del modello
        model = Model(inputs=[input_keypoints, input_opticalflow, input_angles], outputs=output)

        # Compila il modello
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Addestra il modello
        print("\nAddestramento del modello...")
        model.fit([X1, X2, X3], y, epochs=epochs, batch_size=32, validation_split=0.05)
        #model.fit([X1_train, X2_train, X3_train], y, epochs=epochs, batch_size=32, validation_split=0.05)

        # Valuta il modello
        print("\nValutazione del modello...")
        loss, accuracy = model.evaluate([X1_test, X2_test, X3_test], y_test)
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")

        # Salva il modello
        if save_folder is not None:
            model.save(os.path.join(save_folder, "LSTM_Combo3_old_2.h5"))

        return model
    

    @staticmethod
    def combo_3_LSTM_2(X1, X2, X3, y, save_folder = None, epochs=100):
        # Stampa info su X1, X2, X3 e y
        print("\nInfo su X1, X2, X3 e y:")
        print(f"X1 shape: {X1.shape}")
        print(f"X2 shape: {X2.shape}")
        print(f"X3 shape: {X3.shape}")
        print(f"y shape: {y.shape}")

        # ottengo il numero di valori diversi in y
        num_classes = len(np.unique(y))
        print(f"\nNumero di classi: {num_classes}")

        # Genera un array di interi. L'array contiene il 20% dei numeri da 0 a len(X1) (casuali e senza ripetizioni)
        indices = np.random.choice(len(X1), int(len(X1) * 0.2), replace=False)
        # Dividi il dataset in training e test set in base agli indici generati
        X1, X1_test = np.delete(X1, indices, axis=0), X1[indices]
        X2, X2_test = np.delete(X2, indices, axis=0), X2[indices]
        X3, X3_test = np.delete(X3, indices, axis=0), X3[indices]
        y, y_test = np.delete(y, indices, axis=0), y[indices]

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
        print("\nValutazione del modello...")
        loss, accuracy = model.evaluate([X1_test, X2_test, X3_test], y_test)
        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")

        # Salva il modello
        if save_folder is not None:
            model.save(os.path.join(save_folder, "combo_3_LSTM.h5"))

        return model
    

    @staticmethod  
    def LSTM_Combo3_Torch(X1, X2, X3, y, X1_test, X2_test, X3_test, y_test, num_classes, save_folder = None, epochs=20):
        # Stampa info su X1, X2, X3 e y
        print("\nInfo su X1, X2, X3 e y:")
        print(f"X1 shape: {X1.shape}")
        print(f"X2 shape: {X2.shape}")
        print(f"X3 shape: {X3.shape}")
        print(f"y shape: {y.shape}")


class CustomDataset(Dataset):
    def __init__(self, X1, X2, X3, y):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.X3[idx], self.y[idx]

class MultiInputLSTM(nn.Module):
    def __init__(self, input_size_1, input_size_2, input_size_3, hidden_size, num_layers, num_classes):
        super(MultiInputLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size_1, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_size_2, hidden_size, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(input_size_3, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*3, num_classes)
    
    def forward(self, x1, x2, x3):
        # LSTM per il primo ramo
        out1, _ = self.lstm1(x1) # out1: tensor of shape (batch_size, seq_length, hidden_size)
        out1 = out1[:, -1, :] # out1: tensor of shape (batch_size, hidden_size) - prendo l'ultimo output

        # LSTM per il secondo ramo
        out2, _ = self.lstm2(x2)
        out2 = out2[:, -1, :]

        # LSTM per il terzo ramo
        out3, _ = self.lstm3(x3)
        out3 = out3[:, -1, :]

        # Concatenazione dei tre output
        out = torch.cat((out1, out2, out3), 1)

        # Fully connected layer
        out = self.fc(out)

        # sigmoid per probabilità indipendenti
        prob = torch.sigmoid(out)
        return prob
    

def learn_torch(X1, X2, X3, y, X1_test, X2_test, X3_test, y_test, num_classes, save_folder = None):
    # Stampa info su X1, X2, X3 e y
    print("\nInfo su X1, X2, X3 e y:")
    print(f"X1 shape: {X1.shape}")
    print(f"X2 shape: {X2.shape}")
    print(f"X3 shape: {X3.shape}")
    print(f"y shape: {y.shape}")

    # Converto i dati in tensori
    X1 = torch.from_numpy(X1).float()
    X2 = torch.from_numpy(X2).float()
    X3 = torch.from_numpy(X3).float()
    y = torch.from_numpy(y).int().to(torch.long)

    X1_test = torch.from_numpy(X1_test).float()
    X2_test = torch.from_numpy(X2_test).float()
    X3_test = torch.from_numpy(X3_test).float()
    y_test = torch.from_numpy(y_test).int().to(torch.long)

    # Trasformo y da interi a vettori di 0 e 1
    y = torch.nn.functional.one_hot(y, num_classes)
    y_test = torch.nn.functional.one_hot(y_test, num_classes)
    y = y.float()
    y_test = y_test.float()

    # Definisco il modello
    model = MultiInputLSTM(input_size_1=X1.shape[2], input_size_2=X2.shape[2], input_size_3=X3.shape[2], hidden_size=32, num_layers=2, num_classes=num_classes)

    # Definisco la loss e l'ottimizzatore
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Creo il DataLoader
    train_dataset = CustomDataset(X1, X2, X3, y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Addestramento
    for epoch in range(20):
        for X1_batch, X2_batch, X3_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X1_batch, X2_batch, X3_batch)

            # Calcolo della loss
            loss = criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            outputs = model(X1, X2, X3)
            train_predicted = torch.sigmoid(outputs).argmax(dim=1)
            train_labels = y.argmax(dim=1)
            train_accuracy = (train_predicted == train_labels).float().mean().item()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}, Train Accuracy: {train_accuracy}")
    
    # Salva il modello
    if save_folder is not None:
        torch.save(model.state_dict(), os.path.join(save_folder, "combo_3_LSTM_torch.pth"))

    # Valutazione
    with torch.no_grad():
        outputs = model(X1_test, X2_test, X3_test)
        test_predicted = torch.sigmoid(outputs).argmax(dim=1)
        test_labels = y_test.argmax(dim=1)
        test_accuracy = (test_predicted == test_labels).float().mean().item()
        print(f"Test Accuracy: {test_accuracy}")