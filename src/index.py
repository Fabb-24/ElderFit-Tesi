import torch
from torch.utils.data import Dataset, DataLoader
from classification.app import start_application
from graphviz import Digraph
import util
import os
import time
from learning.models_pytorch_old import evaluate_model, CustomDataset
import numpy as np
from learning.models_pytorch_old import create_model
from data.dataset_old import Dataset
from data.data_augmentation import Videos

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


'''if __name__ == "__main__":
    start_application()'''


# stampo i parametri raccolti dai video
params = np.load(os.path.join(util.getParametersPath(), "parameters.npy"), allow_pickle=True).item()
print("\n\n")
for key in params:
    print(f"[{key.replace('_', ' ')}]")
    print(f"    angle_min: {params[key]['angles_min']}")
    print(f"    angle_max: {params[key]['angles_max']}")
print("\n")


'''labels = np.load(os.path.join(util.getDatasetPath(), "labels.npy"))
# genero un grafico a torta che rappresenta la distribuzione delle classi con i nomi delle categorie all'interno
unique, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(10, 6))
plt.pie(counts, labels=unique, autopct='%1.1f%%', startangle=140)
plt.show()'''



#print("\n\nAccuracy: 0.9640\nPrecision: 0.9657\nRecall: 0.9640\nF1-Score: 0.9638\n\n")

'''# estraggo i 3 vettori dai file npy (sono vettori di valori numerici)
accuracies = np.load(os.path.join(util.getModelsPath(), "accuracies3.npy"))
train_losses = np.load(os.path.join(util.getModelsPath(), "train_losses3.npy"))
val_losses = np.load(os.path.join(util.getModelsPath(), "val_losses3.npy"))
y_true = np.load(os.path.join(util.getModelsPath(), "y_test3.npy"))
y_pred = np.load(os.path.join(util.getModelsPath(), "y_pred3.npy"))
# trasformo i vettori da one-hot a interi
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# creo un grafico con accuratezza e val_loss
epochs = range(1, len(accuracies) + 1)  # Genera l'indice delle epoche

# Creare il grafico
plt.figure(figsize=(10, 6))  # Dimensioni della figura (larghezza, altezza)

# Aggiungere le curve con dettagli
#plt.plot(epochs, accuracies, label="Accuracy", color="red", linestyle='-')
plt.plot(epochs, val_losses, label="Training Loss", color="green", linestyle='-.')
plt.plot(epochs, train_losses, label="Validation Loss", color="blue", linestyle='--')

# Aggiungere titolo e etichette
plt.title("Training Progress", fontsize=16)  # Titolo del grafico
plt.xlabel("Epochs", fontsize=14)  # Etichetta asse x
plt.ylabel("Metrics", fontsize=14)  # Etichetta asse y

# Personalizzare l'asse y (opzionale)
plt.ylim(0, 1)  # Cambia i limiti (se necessario)

# Aggiungere griglia
plt.grid(True, linestyle='--', alpha=0.6)

# Legenda
plt.legend(fontsize=12)

# Mostrare il grafico
plt.tight_layout()  # Migliora l'aspetto complessivo
plt.show()

# Creo la matrice di confusione
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()'''


'''content = np.load(os.path.join(util.getParametersPath(), "parameters.npy"), allow_pickle=True).item()
print(content)

content['arms_up']['angles_max'] = 46.59411557814686
np.save(os.path.join(util.getParametersPath(), "parameters.npy"), content)'''

# Valuto il modello
'''model = util.get_pytorch_model(os.path.join(util.getModelsPath(), "LSTM_Combo3.pth"))
X1_test, X2_test, X3_test, y_test, num_classes = util.get_dataset("test")
y_test = torch.from_numpy(y_test).int().to(torch.long)
y_test = torch.nn.functional.one_hot(y_test, num_classes=num_classes).float()
print("\n\n")
evaluate_model(model, X1_test, X2_test, X3_test, y_test, "cpu")
print("\n\n")'''

'''def create_dataset():
    Dataset().create()


def learning():
    X1, X2, X3, y, num_classes = util.get_dataset("train")
    X1_test, X2_test, X3_test, y_test, _ = util.get_dataset("test")
    create_model(X1, X2, X3, y, X1_test, X2_test, X3_test, y_test, num_classes)

create_dataset()
learning()'''