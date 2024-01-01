import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from src.utils import get_transform, get_classes
from PIL import Image


class CustomDataset(Dataset):

    def __init__(self, root_dir, split='train',BATCH_SIZE=64):
        # self.data_dir = '/mnt/e/Oulu University/Semester 1/Period 2/Deep Learning/Project/Dataset/miniImageNet/train/train/'

        self.root_dir = root_dir
        self.transform = get_transform()
        self.BATCH_SIZE = BATCH_SIZE

        # get list of classes in our dataset
        self.classes = sorted(get_classes(self.root_dir))

    # def get_data(self, split='train'):
        # Taking all file paths and labels
        all_files = []
        all_labels = []
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)
            files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
            all_files.extend(files)
            all_labels.extend([label] * len(files))
            # print(files,class_name)

        # spliting our dataset into train, validation and test
        train_files, test_files, train_labels, test_labels = train_test_split(
            all_files, all_labels, test_size=0.3, random_state=42
        )
        valid_files, test_files, valid_labels, test_labels = train_test_split(
            test_files, test_labels, test_size=0.5, random_state=42
        )

        # set the data and labels based on the split
        if split == 'train':
            self.data = train_files
            self.labels = train_labels
        elif split == 'valid':
            self.data = valid_files
            self.labels = valid_labels
        elif split == 'test':
            self.data = test_files
            self.labels = test_labels
        else:
            raise ValueError("Invalid split, Use 'train', 'valid' or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label

    def load_data(self):
        train_dataset = self.get_data(split='train')
        valid_dataset = self.get_data(split='valid')
        test_dataset = self.get_data(split='test')
        print(f"Number of images in training set: {len(train_dataset)}")
        print(f"Number of images in validation set: {len(valid_dataset)}")
        print(f"Number of images in test set: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        return train_loader, val_loader, test_loader
