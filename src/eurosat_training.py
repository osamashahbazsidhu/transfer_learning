import torch.nn as nn
from torchvision import models
import timm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EuroSatModelTraining:
    def __init__(self, device):
        self.device = device
        # self.model = None
        pass

    def get_model(self, model_name="resnet18", classes=None):
        if model_name == "resnet18":
            resnet = models.resnet18(pretrained=True)
            for param in resnet.parameters():
                param.requires_grad = False
            num_features = resnet.fc.in_features
            resnet.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, classes)
            ).to(self.device)
            return resnet

    def get_optimizer(self, optimizer_name, model, learning_rate):
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        return optimizer

    def get_schedular(self, schedular_name, optimizer):
        if schedular_name == "StepLR":
            scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        elif schedular_name == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
        else:
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        return scheduler

    def train_model(self, model_name, optimizer_name, schedular_name, train_loader, test_loader,
                    num_epochs=25, learning_rate=0.001, classes=None, model_path=None):
        criterion = nn.CrossEntropyLoss()
        model = self.get_model(model_name, classes=classes)
        best_val_accuracy = 0.0
        weights_path = model_path
        state_dict = torch.load(weights_path)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        # optimizer = torch.optim.RMSprop(fine_tuned_resnet18.resnet18.fc.parameters(), lr=0.001, alpha=0.9)
        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer(optimizer_name, model, learning_rate)
        schedular = self.get_schedular(schedular_name, optimizer)


        train_acc_history = []
        test_acc_history = []

        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            total_train = 0
            correct_train = 0
            with torch.no_grad():
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs_train = model(inputs)
                    _, predicted_train = torch.max(outputs_train.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted_train == labels).sum().item()

                accuracy_train = correct_train / total_train
                train_acc_history.append(accuracy_train)
            # schedular.step()
            # schedular.step(val_loss.item())

            print(
                f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Train Accuracy: {accuracy_train * 100:.2f}')

            # Test the model
            model.eval()
            total_test = 0
            correct_test = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs_test = model(inputs)
                    _, predicted_test = torch.max(outputs_test.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted_test == labels).sum().item()

            accuracy_test = correct_test / total_test
            test_acc_history.append(accuracy_test)  # Store testing accuracy
            print(f'Test Accuracy: {accuracy_test * 100:.2f}%')

        torch.save(model.state_dict(), f'results/transfer_learning{model_name}_resnet18_model.pth')
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy')
        plt.plot(range(1, num_epochs + 1), test_acc_history, label='Testing Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training, Validation, and Testing Accuracy Curves')
        plt.legend()
        plt.savefig(f'results/transfer_learning{model_name}_{optimizer_name}_{schedular_name}.png')
        plt.show()
