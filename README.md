# Neural Network with One Hidden Layer

This page demostrates some sample code for neural network with one hidden layer. Two Python files has been shared.

`neural_network.py` is written from scratch without any Machine Learning libraries such as `TensorFlow`.

`neural_network_keras.py` uses the capabilities of `TensorFlow` tools.

## Results

### Gradient Descend

`neural_network.py` manages to decrease the cost and it converges as seen below.

<img src="https://user-images.githubusercontent.com/22200109/210360354-6ae6cbbf-19c8-48a6-a23e-87ff20e79ff1.png" width="500">

### Accuracy

Using the same train and and test sets, the accuracy scores of both version:

|   | Accuracy |
| ------------- | ------------- |
| `neural_network.py`  | 0.7942  |
| `neural_network_keras.py`  | 0.7846  |
