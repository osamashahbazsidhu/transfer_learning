import os
from torch.utils.data import Dataset, DataLoader
from src.utils import get_transform, get_classes
from PIL import Image


class CustomEuroSatDataset(Dataset):

    def __init__(self, root_dir, split='train', BATCH_SIZE=64):
        self.root_dir = root_dir
        self.transform = get_transform()
        self.BATCH_SIZE = BATCH_SIZE
        self.classes = sorted(get_classes(self.root_dir))

        all_files = []
        all_labels = []
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
            all_files.extend(files)
            all_labels.extend([label] * len(files))
        if split == 'train':
            self.data = all_files
            self.labels = all_labels
        elif split == 'valid':
            self.data = all_files
            self.labels = all_labels
        elif split == 'test':
            self.data = all_files
            self.labels = all_labels
        else:
            raise ValueError("Invalid split, Use 'train', 'valid' or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
