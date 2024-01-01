import torch.nn as nn
from torchvision import models
import timm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ModelTraining:
    def __init__(self,device):
        self.device = device
        # self.model = None
        pass

    def get_model(self, model_name="resnet18",classes=None):
        if model_name == "resnet18":
            resnet = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
            num_features = resnet.fc.in_features
            # resnet.fc = nn.Linear(num_features, classes)
            resnet.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(resnet.fc.in_features, classes)
            ).to('cuda')
            resnet = resnet.to(self.device)
            # self.model = resnet
            return resnet

        elif model_name=="vgg16":
            vgg = models.vgg16(pretrained=True)
            vgg.classifier[-1] = nn.Linear(vgg.classifier[-1].in_features, classes)
            # self.model = vgg
            return vgg

        elif model_name =="vit":
            vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            vit_model.head = nn.Linear(vit_model.head.in_features, classes)
            # self.model = vit_model

            return vit_model

        elif model_name =="vgg19":
            vgg19 = models.vgg19(pretrained=True)
            vgg19.classifier[-1] = nn.Linear(vgg19.classifier[-1].in_features, classes)
            # self.model = vgg19
            return vgg19

    def get_optimizer(self, optimizer_name,model,learning_rate):
        if optimizer_name =="Adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name=="SGD":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        return optimizer

    def get_schedular(self,schedular_name,optimizer):
        if schedular_name == "StepLR":
            scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        elif schedular_name == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
        else:
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        return  scheduler


    def train_model(self, model_name, optimizer_name, schedular_name, train_loader, val_loader,test_loader, num_epochs=25,learning_rate=0.001,classes=None):
        criterion = nn.CrossEntropyLoss()
        model = self.get_model(model_name,classes=classes)
        optimizer= self.get_optimizer(optimizer_name,model,learning_rate)
        schedular = self.get_schedular(schedular_name,optimizer)
        best_val_accuracy = 0.0

        train_acc_history = []
        val_acc_history = []
        test_acc_history = []

        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validate the model
            model.eval()
            total_val = 0
            correct_val = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    val_loss = criterion(outputs, labels)
                    _, predicted_val = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted_val == labels).sum().item()

                accuracy_val = correct_val / total_val
                val_acc_history.append(accuracy_val)  # Store validation accuracy
                print(accuracy_val)

            # Store training accuracy
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
                train_acc_history.append(accuracy_train)  # Store training accuracy
            # scheduler.step()
            schedular.step(val_loss.item())

            print(
                f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {accuracy_val * 100:.2f}, Train Accuracy: {accuracy_train * 100:.2f}')

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

        torch.save(model.state_dict(), 'results/resnet18_model.pth')
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy')
        plt.plot(range(1, num_epochs + 1), val_acc_history, label='Validation Accuracy')
        plt.plot(range(1, num_epochs + 1), test_acc_history, label='Testing Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training, Validation, and Testing Accuracy Curves')
        plt.legend()
        plt.savefig(f'results/{model_name}_{optimizer_name}_{schedular_name}.png')
        plt.show()
