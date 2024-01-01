import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
def get_transform():
    transform = transforms.Compose([
        # transforms.RandomRotation(10),      # rotate 10 degrees
        # transforms.RandomHorizontalFlip(),  # reverse 50% of images
        # transforms.Resize(224),             # resize shortest side to 224 pixels
        # transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],  # recommended values for imagenet
                             [0.5, 0.5, 0.5])
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