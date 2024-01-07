import os
import random
import shutil
from sklearn.model_selection import train_test_split


class EuroSatData:
    def __init__(self, dataset_dir, train_set_dir, test_set_dir, classes):
        self.dataset_dir = dataset_dir
        self.train_set_dir = train_set_dir
        self.test_set_dir = test_set_dir
        self.classes = classes

    def data_extractor(self):
        if os.path.exists(self.train_set_dir):
            shutil.rmtree(self.train_set_dir)
            print(f"Directory '{self.train_set_dir}' deleted.")
        else:
            print(f"Directory '{self.train_set_dir}' does not exist.")

        if os.path.exists(self.test_set_dir):
            shutil.rmtree(self.test_set_dir)
            print(f"Directory '{self.test_set_dir}' deleted.")
        else:
            print(f"Directory '{self.test_set_dir}' does not exist.")

        class_folders = os.listdir(self.dataset_dir)
        selected_classes = random.sample(class_folders, 5)
        print("Selected classes:", selected_classes)
        selected_indices = {class_folder: [] for class_folder in selected_classes}
        X_train = []
        X_test = []
        Y_train = []
        Y_test = []
        for class_folder in selected_classes:
            all_labels = []
            class_path = os.path.join(self.dataset_dir, class_folder)
            all_images = os.listdir(class_path)
            selected_indices[class_folder] = random.sample(all_images, 20)
            all_labels.extend([class_folder] * len(selected_indices[class_folder]))
            train_x, test_x, train_y, test_y = train_test_split(
                selected_indices[class_folder], all_labels, test_size=0.75, random_state=42
            )
            X_train.extend(train_x)
            X_test.extend(test_x)
            Y_train.extend(train_y)
            Y_test.extend(test_y)

        os.makedirs(self.train_set_dir, exist_ok=True)
        for i in range(len(X_train)):
            temp_dir = os.path.join(self.dataset_dir, Y_train[i])
            source = os.path.join(temp_dir, X_train[i])
            destination_path = os.path.join(self.train_set_dir, Y_train[i])
            os.makedirs(destination_path, exist_ok=True)
            destination = os.path.join(destination_path, X_train[i])

            shutil.copyfile(source, destination)

        for i in range(len(X_test)):
            temp_dir = os.path.join(self.dataset_dir, Y_test[i])
            source = os.path.join(temp_dir, X_test[i])
            destination_path = os.path.join(self.test_set_dir, Y_test[i])
            os.makedirs(destination_path, exist_ok=True)
            destination = os.path.join(destination_path, X_test[i])
            shutil.copyfile(source, destination)

        return selected_classes
