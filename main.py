import json
import torch
from src.data_loader import CustomDataset
from src.eurosat_dataloader import CustomEuroSatDataset
from src.data_preparation import EuroSatData
from src.utils import load_data, get_classes, load_euro_sat_data
from src.train import ModelTraining
from src.eurosat_training import EuroSatModelTraining

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device:{device}")


def load_config(config_file_path):
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)
    return config


def main():
    config = load_config('config.json')
    BATCH_SIZE = config["batch_size"]
    model_name = config["model"]["name"]
    optimizer_name = config["model"]["optimizer"]
    scheduler_name = config["model"]["scheduler"]
    learning_rate = config["model"]["learning_rate"]
    num_epochs = config["model"]["num_epochs"]

    data_dir = config["dataset"]["data_dir"]
    print(BATCH_SIZE,model_name,optimizer_name,scheduler_name,learning_rate,num_epochs,data_dir)
    classes = len(sorted(get_classes(data_dir)))
    train_loader, val_loader, test_loader = load_data(data_dir, BATCH_SIZE, CustomDataset)
    model = ModelTraining(device)
    model_path = model.train_model(model_name=model_name, optimizer_name=optimizer_name, schedular_name=scheduler_name,
                                   train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                   num_epochs=num_epochs,
                                   learning_rate=learning_rate, classes=classes)

    dataset_dir = config["dataset"]["euro_sat"]["dataset_dir"]
    train_set_dir = config["dataset"]["euro_sat"]["train_set_dir"]
    test_set_dir = config["dataset"]["euro_sat"]["test_set_dir"]
    classes = config["dataset"]["euro_sat"]["classes"]
    eurosat_data = EuroSatData(dataset_dir, train_set_dir, test_set_dir, classes)
    print(eurosat_data.data_extractor())
    train_loader, test_loader = load_euro_sat_data(train_set_dir, test_set_dir, BATCH_SIZE, CustomEuroSatDataset)
    model = EuroSatModelTraining(device)
    model.train_model(model_name=model_name, optimizer_name=optimizer_name, schedular_name="StepLR",
                      train_loader=train_loader, test_loader=test_loader, num_epochs=25,
                      learning_rate=0.001, classes=classes, model_path=model_path)


if __name__ == '__main__':
    main()
