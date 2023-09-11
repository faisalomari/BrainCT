# Brain Disease Detection Project

## Overview

This project aims to detect various brain diseases, including Epidural, Subdural, Intraventricular, Intraparenchymal, Subarachnoid, No_Hemorrhage, and Fracture_Yes_No, using medical images. The dataset used for this project can be found [here](https://gle.com/datasets/vbookshelf/computed-tomography-ct-images).

## Data Augmentation

Data augmentation is a critical step in training machine learning models. In this project, the data augmentation script is used to increase the size and diversity of the dataset. The script utilizes the `ImageDataGenerator` class from TensorFlow's Keras library to apply various transformations to the images, such as rotation, zooming, and flipping. This process helps in improving the model's generalization and robustness.

## Model Architectures

Several machine learning and deep learning models have been implemented and trained for disease detection:

### Artificial Neural Network (ANN)

The ANN model consists of multiple layers, including dense layers with ReLU activation functions. It's trained to predict whether a given medical image contains signs of a specific brain disease.

### Convolutional Neural Network (CNN)

A CNN is a specialized deep learning model designed for image-related tasks. It's used to extract hierarchical features from medical images and make predictions based on these features.

### Long Short-Term Memory (LSTM)

LSTM is a type of recurrent neural network (RNN) that can be used for sequential data. In this project, LSTM is applied to sequences of data extracted from medical images to make predictions.

### Support Vector Machine (SVM)

SVM is a traditional machine learning algorithm used for classification tasks. It's applied to extract features from medical images and make disease predictions.

### Logistic Regression

Logistic Regression is another traditional machine learning technique used for binary classification tasks. It's applied to medical image features to predict the presence of specific brain diseases.

## Model Evaluation

The trained models are evaluated using various metrics, including accuracy and confusion matrices. These metrics help assess the performance of each model in disease detection.

## Execution Time

The total execution time of the project, from data preprocessing to model training and evaluation, is recorded for reference.

## Repository Structure

The project repository is organized as follows:

- `src/data_preproccessing/generate_images.py`: Script for data augmentation.
- `results/`: Directory containing graphs and results.
- `Extraction.py`: Module for feature extraction.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary dependencies.
3. Run the data augmentation script if needed.
4. Train and evaluate the disease detection models using the provided scripts.

## Results

The project results, including accuracy graphs and confusion matrices, can be found in the `results/` directory. These visuals provide insights into the performance of each model.

## Conclusion

This project demonstrates the use of various machine learning and deep learning models for brain disease detection using medical images. The combination of data augmentation and different model architectures allows for accurate and reliable disease prediction.

For further details and specific model performance, please refer to the individual model directories and evaluation results.

---

**Note**: Please customize this readme with specific details, images, and links related to your project. You can also include information about your dataset, data preprocessing, and any additional insights or findings.
