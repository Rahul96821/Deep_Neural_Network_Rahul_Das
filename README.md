# Deep Neural Network from Scratch

This project implements a **Deep Neural Network (DNN)** from scratch using **NumPy**.
It covers every step of building and training a neural network, including:

* Parameter initialization
* Forward propagation (linear + activation functions)
* Cost computation
* Backward propagation (gradients for each layer)
* Parameters update with gradient descent

The goal is to understand **how deep learning works under the hood** without using high-level frameworks like TensorFlow or PyTorch.

---

## 🚀 Features

* Build neural networks with **arbitrary number of layers**
* Supports **ReLU** and **Sigmoid** activations
* Implements **binary classification** (e.g., cat vs non-cat)
* Fully vectorized using **NumPy** (no loops where possible)
* Modular code structure (easy to extend for new activations, cost functions, etc.)

---

## 📂 Project Structure

```
├── dnn_utils.py           # Helper functions (sigmoid, relu, derivatives, etc.)
├── testCases.py           # Predefined test cases for validation
├── public_tests.py        # Test utilities
├── model_notebook.ipynb   # Jupyter notebook with full implementation
├── README.md              # Project documentation
```

---

## ⚙️ How It Works

1. **Initialize parameters** (`initialize_parameters`, `initialize_parameters_deep`)
2. **Forward propagation**

   * `linear_forward`
   * `linear_activation_forward`
   * `L_model_forward`
3. **Compute cost** (`compute_cost`)
4. **Backward propagation**

   * `linear_backward`
   * `linear_activation_backward`
   * `L_model_backward`
5. **Update parameters** (`update_parameters`)

---

## 🧪 Example Usage

```python
# Example with a 3-layer network
layer_dims = [4, 5, 3, 1]  # 4 inputs, 2 hidden layers, 1 output

# Initialize
parameters = initialize_parameters_deep(layer_dims)

# Forward pass
AL, caches = L_model_forward(X, parameters)

# Compute cost
cost = compute_cost(AL, Y)

# Backward pass
grads = L_model_backward(AL, Y, caches)

# Update parameters
parameters = update_parameters(parameters, grads, learning_rate=0.0075)
```

---

## 📊 Training Results

After training on a binary classification dataset (e.g., cat vs non-cat),
the network typically achieves **>95% training accuracy**.

---

## 🛠️ Requirements

* Python 3.7+
* NumPy
* Matplotlib
* h5py (if using datasets in `.h5` format)


## 📚 Learning Objective

This project is part of my journey to **understand deep learning fundamentals**.
Instead of relying on high-level libraries, I implemented all building blocks manually.

