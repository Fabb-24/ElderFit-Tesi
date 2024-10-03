import torch
from torch.utils.data import Dataset, DataLoader
from classification.app import start_application
from graphviz import Digraph
import util
import os
from learning.models_pytorch import evaluate_model, CustomDataset
import numpy as np
from learning.models_pytorch import create_model
from data.dataset import Dataset
from data.dataAugmentation import Videos


if __name__ == "__main__":
    start_application()

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