# Siamese Network for Fashion-MNIST Classification

This project implements a Siamese Neural Network using Keras and TensorFlow for classifying images in the Fashion-MNIST dataset. A Siamese network is designed to find the similarity between two input images by learning a feature representation and computing the Euclidean distance between them. The network is trained using contrastive loss to minimize the distance between similar images and maximize it between dissimilar ones.

## Project Overview

The key features of this project include:

- **Dataset**: Fashion-MNIST Dataset, consisting of 70,000 grayscale images (60,000 training and 10,000 test) across 10 different classes.
- **Network Architecture**: The base network is built with convolutional layers followed by dense layers, using `relu` and `tanh` activations to learn feature embeddings.
- **Distance Calculation**: The model computes the Euclidean distance between two feature vectors and uses a contrastive loss function for training.
- **Optimization**: The model is optimized using the RMSProp optimizer.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Model Architecture

The base network consists of:

1. Convolutional layers for feature extraction:
   - 32 filters with a 3x3 kernel size and ReLU activation.
   - Average pooling with a 2x2 window.
   - 64 filters with a 3x3 kernel size and Tanh activation.
   - Max pooling with a 2x2 window.
2. Fully connected dense layers:
   - Flattened output followed by a dense layer with 128 units and Tanh activation.
   - Dropout layers to reduce overfitting.
   - Final dense layer with 10 units and Tanh activation for output.

## How to Run

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Run the script to train the model:

   ```bash
   python siamese_fashion_mnist.py

