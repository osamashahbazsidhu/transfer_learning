# Transfer Learning Image Based Classification

## Overview

This project implements transfer learning for image classification using various pre-trained models. Users can fine-tune the models on their own dataset by configuring the settings in the provided JSON file. The application supports models such as ResNet18, VGG16, ViT, VGG19, MobileNet, and EfficientNet.

## Prerequisites

Before running the application, make sure to install the required Python packages by executing the following command:

```bash
pip install -r requirements.txt
```

## Configuration

Configure the training process by editing the `config.json` file. The file contains the following parameters:

- **batch_size**: The batch size for training.

- **model**: Configuration for the selected model.
  - **name**: Choose from [resnet18, vgg16, vit, vgg19, mobilenet, effnet].
  - **optimizer**: Choose either "Adam" or "SGD".
  - **scheduler**: Choose either "StepLR" or "ReduceLROnPlateau".
  - **learning_rate**: The initial learning rate for the optimizer.
  - **num_epochs**: The number of training epochs.

- **dataset**: Configuration for the dataset.
  - **data_dir**: The root directory of the training dataset.
  - **euro_sat**: Configuration specific to the EuroSAT dataset.
    - **dataset_dir**: The directory containing EuroSAT dataset.
    - **train_set_dir**: The directory for the training set within EuroSAT.
    - **test_set_dir**: The directory for the test set within EuroSAT.
    - **classes**: The number of classes in the EuroSAT dataset.

## Datasets

This application uses the following datasets:

- **Pretrained Dataset**: MiniImagenet
  - Description: MiniImagenet is used for pretraining the selected models.
  - [MiniImagenet Dataset](https://drive.google.com/drive/folders/17a09kkqVivZQFggCw9I_YboJ23tcexNM)

- **Fine-tuning and Evaluation Dataset**: EuroSAT
  - Description: EuroSAT is used for fine-tuning the pretrained models and evaluating the performance.
  - [EuroSAT Dataset](https://github.com/phelber/EuroSAT?tab=readme-ov-file)

## Running the Application

To train the model with the specified configuration, run the following command:

```bash
python main.py
```
This will start the training process using the configured settings. Adjust the config.json file to customize the training for your specific requirements.
