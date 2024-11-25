import os
import numpy as np
from data.dataset import Dataset
from data.data_augmentation import Videos
#from learning.models_keras import create_model
from learning.models_pytorch import create_model, train_best_model
import util
from classification.app import start_application


def process_videos():
    Videos(util.getVideoPath()).process_videos()


def create_dataset():
    Dataset().create()


'''def learning():
    X1, X2, X3, y, num_classes = util.get_dataset("train")
    X1_test, X2_test, X3_test, y_test, _ = util.get_dataset("test")
    create_model(X1, X2, X3, y, X1_test, X2_test, X3_test, y_test, num_classes)'''

def learning():
    X1 = np.load(os.path.join(util.getDatasetPath(), "keypoints.npy"))
    X2 = np.load(os.path.join(util.getDatasetPath(), "angles.npy"))
    y = np.load(os.path.join(util.getDatasetPath(), "labels.npy"))
    num_classes = len(np.unique(y))
    categories = np.load(os.path.join(util.getDatasetPath(), "categories.npy"))
    y = np.array([list(categories).index(label) for label in y])

    X1_test = np.load(os.path.join(util.getDatasetTestPath(), "keypoints.npy"))
    X2_test = np.load(os.path.join(util.getDatasetTestPath(), "angles.npy"))
    y_test = np.load(os.path.join(util.getDatasetTestPath(), "labels.npy"))
    categories_test = np.load(os.path.join(util.getDatasetTestPath(), "categories.npy"))
    y_test = np.array([list(categories_test).index(label) for label in y_test])

    # rimuovo le righe in X1 e X2 che corrispondono alla label 6 (X1 e X2 hanno un particolare ordine che corrisponde a quello delle labels)
    # se la riga in y è 6, rimuovo quella riga anche in X1 e X2
    '''to_remove = []
    for i in range(len(y)):
        if y[i] == 6:
            to_remove.append(i)
    X1 = np.delete(X1, to_remove, axis=0)
    X2 = np.delete(X2, to_remove, axis=0)
    y = np.delete(y, to_remove)

    to_remove = []
    for i in range(len(y_test)):
        if y_test[i] == 6:
            to_remove.append(i)
    X1_test = np.delete(X1_test, to_remove, axis=0)
    X2_test = np.delete(X2_test, to_remove, axis=0)
    y_test = np.delete(y_test, to_remove)

    num_classes -= 1'''

    #create_model(X1, X2, y, X1_test, X2_test, y_test, num_classes)

    best_params = np.load(os.path.join(util.getModelsPath(), "best_params_full.npy"), allow_pickle=True).item()
    train_best_model(best_params, X1, X2, y, X1_test, X2_test, y_test, num_classes, util.getModelsPath())


if __name__ == "__main__":
    while True:
        print("\nMenu")
        print("1. Process Videos")
        print("2. Create Dataset")
        print("3. Create Model")
        print("4. Start Application")
        print("5. Exit")

        choice = int(input("Enter your choice: "))

        if choice == 1:
            process_videos()
        elif choice == 2:
            create_dataset()
        elif choice == 3:
            learning()
        elif choice == 4:
            start_application()
            print("Application started")
        elif choice == 5:
            break
        else:
            print("Invalid choice")