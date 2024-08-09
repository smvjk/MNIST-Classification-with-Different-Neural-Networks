# MNIST Classification with Different Neural Networks

This repository contains a PyTorch implementation of various neural network models trained on the MNIST dataset. The models include a Perceptron, a Deep Neural Network, and a Convolutional Neural Network (CNN). The project aims to demonstrate the training and evaluation of these models on the MNIST dataset.

## Table of Contents
- [Introduction](#introduction)
- [Models](#models)
- [Results](#results)
- [Requirements](#requirements)

## Introduction
The MNIST dataset consists of 28x28 grayscale images of handwritten digits, labeled from 0 to 9. It is a standard benchmark for machine learning models. In this project, we compare the performance of three different neural network architectures:
- **Perceptron**: A simple feedforward neural network.
- **Deep Neural Network**: A slightly more complex feedforward network.
- **Convolutional Neural Network (CNN)**: A network specifically designed for image data.

## Models
### 1. **Perceptron**
- **Architecture**: A single hidden layer with 128 neurons.
- **Activation Function**: ReLU.
- **Output**: 10 neurons with Log-Softmax.

### 2. **Deep Neural Network**
- **Architecture**: Two fully connected layers with 20 and 10 neurons, respectively.
- **Activation Function**: ReLU.
- **Output**: 10 neurons with Log-Softmax.

### 3. **Convolutional Neural Network (CNN)**
- **Architecture**: Two convolutional layers followed by max-pooling, dropout, and two fully connected layers.
- **Activation Function**: ReLU.
- **Output**: 10 neurons with Log-Softmax.

##Results
After training the models for 50 epochs, the following results were observed on the test set:
-**Perceptron**: Achieved an accuracy of ~98.15%.
-**Deep Neural Network**: Achieved an accuracy of ~95.76%.
-**Convolutional Neural Network (CNN)**: Achieved an accuracy of ~96.59%.
Confusion matrices for each model are generated at the end of the script, visualizing the performance of each model on the test data.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn
