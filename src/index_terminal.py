from data.dataset import Dataset
from data.dataAugmentation import Videos
#from learning.models_keras import create_model
from learning.models_pytorch import create_model
import util
from classification.app import start_application


def process_videos():
    Videos(util.getVideoPath()).process_videos()


def create_dataset():
    Dataset().create()


def learning():
    X1, X2, X3, y, num_classes = util.get_dataset("train")
    X1_test, X2_test, X3_test, y_test, _ = util.get_dataset("test")
    create_model(X1, X2, X3, y, X1_test, X2_test, X3_test, y_test, num_classes)


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