import torch
from src.data_loader import CustomDataset
from src.utils import load_data, get_classes
from src.train import ModelTraining
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")
def main():
    BATCH_SIZE = 64
    data_dir = 'train/train'

    classes = len(sorted(get_classes(data_dir)))
    train_loader, val_loader, test_loader = load_data(data_dir, BATCH_SIZE, CustomDataset)
    model = ModelTraining(device)
    model.train_model(model_name="resnet18",optimizer_name="SGD",schedular_name="StepLR",
                      train_loader=train_loader, val_loader=val_loader,test_loader=test_loader, num_epochs=15,
                      learning_rate=0.001,classes=classes)


if __name__ == '__main__':
    main()

