HRNetworkGO

HRNetworkGO is a machine learning project that uses a neural network to predict hand writing images from the MNIST dataset. The neural network is implemented from scratch using the Go programming language.

Overview

The MNIST dataset consists of 70,000 images of hand-written digits, each of which is 28x28 pixels in size. HRNetworkGO uses this dataset to train a neural network that can then be used to classify new images as one of the 10 digits (0-9). The neural network uses backpropagation to adjust its weights and biases in order to minimize the prediction error.

Features
- Implemented in Go for performance and simplicity

- Uses backpropagation for training the neural network

- Feed forward implementation for making predictions

- Trains on the MNIST dataset for accurate predictions


Usage

To use HRNetworkGo, you will need to have Go installed on your machine. Once Go is installed, you can clone the repository to your local machine using the following command:

- $ git clone https://github.com/jaceg18/HRNetworkGo.git

Then, navigate to the project directory and build the program using the following command:

- $ go build

This will create an executable file named HRNetworkGo in the project directory. You can then run the program using the following command:

- $ ./HRNetworkGo

The program will output the accuracy of the neural network on the MNIST test set.


Contributions

Contributions are welcome! If you would like to contribute to HRNetworkGO, please fork the repository and create a pull request with your changes.
