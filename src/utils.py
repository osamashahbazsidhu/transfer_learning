import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader


def get_transform():
    transform = transforms.Compose([
        # transforms.Resize((224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5],
        #                     [0.5, 0.5, 0.5]),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

    ])
    return transform


def get_classes(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def load_data(root_dir, batch_size, dataset_instance):
    train_dataset = dataset_instance(root_dir=root_dir, split='train', BATCH_SIZE=batch_size)
    valid_dataset = dataset_instance(root_dir=root_dir, split='valid', BATCH_SIZE=batch_size)
    test_dataset = dataset_instance(root_dir=root_dir, split='test', BATCH_SIZE=batch_size)
    print(f"Number of images in training set: {len(train_dataset)}")
    print(f"Number of images in validation set: {len(valid_dataset)}")
    print(f"Number of images in test set: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


def load_euro_sat_data(train_data_dir, test_data_dir, batch_size, dataset_instance):
    train_dataset = dataset_instance(root_dir=train_data_dir, split='train', BATCH_SIZE=batch_size)
    test_dataset = dataset_instance(root_dir=test_data_dir, split='test', BATCH_SIZE=batch_size)
    print(len(train_dataset))
    print(len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
