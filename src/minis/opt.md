---
layout: post
title:  Optimization Playground
parent: Minis
nav_order: 6
---

## Optimization Playground: Gradient Descent + Neural Network Optimization

If you haven't learned the optimization basics yet, start with the [Gradient Descent Theory lesson](https://astarryknight.github.io/ai-ml/src/theory/grad_desc.html).

**Objective:** Build intuition for optimization by training:
1. A linear regression model with simple gradient descent (square trick idea)
2. A tiny neural network with backpropagation

Everything below is designed to run directly in Google Colab.

----

### Part 1: Simple Gradient Descent (Linear Regression)

We will reuse the same idea from the theory lesson: start with random parameters, make predictions, compute error, and nudge parameters in the direction that reduces error.

**Setup + Data**

```python
import numpy as np
import matplotlib.pyplot as plt
import random

# Campaign fundraising toy dataset
features = np.array([1, 2, 3, 5, 6, 7, 8], dtype=np.float64)
labels = np.array([155, 197, 244, 356, 409, 448, 500], dtype=np.float64)

plt.figure(figsize=(5, 4))
plt.scatter(features, labels)
plt.title("Campaign Fundraising")
plt.xlabel("Number of Donations")
plt.ylabel("Funds Raised")
plt.show()
```

**Gradient Descent Utilities**

```python
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Square trick style update (single point)
def square_trick(base_funds, funds_per_dono, num_donos, funds, learning_rate):
    pred = base_funds + funds_per_dono * num_donos
    funds_per_dono += learning_rate * num_donos * (funds - pred)
    base_funds += learning_rate * (funds - pred)
    return base_funds, funds_per_dono


def train_linear(features, labels, learning_rate=0.01, epochs=2000, seed=7):
    random.seed(seed)
    base_funds = random.random()
    funds_per_dono = random.random()
    errors = []

    for _ in range(epochs):
        preds = base_funds + funds_per_dono * features
        errors.append(rmse(labels, preds))

        i = random.randint(0, len(features) - 1)
        base_funds, funds_per_dono = square_trick(
            base_funds,
            funds_per_dono,
            features[i],
            labels[i],
            learning_rate,
        )

    return base_funds, funds_per_dono, errors
```

**Run and Interact**

```python
# Try changing these and re-running this cell
learning_rate = 0.01
epochs = 2000
seed = 7

base_funds, funds_per_dono, errors = train_linear(
    features, labels, learning_rate=learning_rate, epochs=epochs, seed=seed
)

preds = base_funds + funds_per_dono * features

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Fitted line
ax[0].scatter(features, labels, label='Data')
ax[0].plot(features, preds, color='red', label='Fitted Line')
ax[0].set_title('Linear Regression Fit')
ax[0].set_xlabel('Number of Donations')
ax[0].set_ylabel('Funds Raised')
ax[0].legend()

# Error curve
ax[1].plot(errors)
ax[1].set_title('RMSE During Training')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('RMSE')

plt.show()

print(f"funds_per_dono (slope): {funds_per_dono:.4f}")
print(f"base_funds (intercept): {base_funds:.4f}")
print(f"final RMSE: {errors[-1]:.4f}")
```

What to try:
- Increase `learning_rate` to `0.1` and observe instability.
- Decrease it to `0.001` and observe slower convergence.
- Change `epochs` and compare final RMSE.

----

### Part 2: Neural Network Optimization (2-Layer Net)

Now we do the same optimization loop for a small neural net:
- Forward pass to get predictions
- Compute loss
- Backpropagation to compute gradients
- Gradient descent update on weights

We will classify a nonlinear 2D dataset.

**Create Dataset**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
y = y.reshape(-1, 1)

plt.figure(figsize=(5, 4))
plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap='coolwarm', s=25)
plt.title('Two Moons Dataset')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
```

**Model + Training Code (NumPy Only)**

```python
# Activations
def tanh(z):
    return np.tanh(z)

def tanh_grad(a):
    return 1 - a**2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary cross-entropy
def bce_loss(y_true, y_pred, eps=1e-8):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def init_params(input_dim=2, hidden_dim=8, output_dim=1, seed=42):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.5, size=(input_dim, hidden_dim))
    b1 = np.zeros((1, hidden_dim))
    W2 = rng.normal(0, 0.5, size=(hidden_dim, output_dim))
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = tanh(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)
    return z1, a1, z2, y_hat


def train_nn(X, y, hidden_dim=8, lr=0.1, epochs=3000, seed=42):
    W1, b1, W2, b2 = init_params(2, hidden_dim, 1, seed=seed)
    losses, accs = [], []
    m = X.shape[0]

    for _ in range(epochs):
        z1, a1, z2, y_hat = forward(X, W1, b1, W2, b2)

        # Loss + accuracy
        loss = bce_loss(y, y_hat)
        y_pred = (y_hat >= 0.5).astype(int)
        acc = np.mean(y_pred == y)
        losses.append(loss)
        accs.append(acc)

        # Backprop
        dz2 = (y_hat - y) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * tanh_grad(a1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Gradient descent update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return W1, b1, W2, b2, losses, accs
```

**Run and Interact**

```python
# Try changing these values and re-running
hidden_dim = 8
learning_rate = 0.1
epochs = 3000
seed = 42

W1, b1, W2, b2, losses, accs = train_nn(
    X, y,
    hidden_dim=hidden_dim,
    lr=learning_rate,
    epochs=epochs,
    seed=seed
)

# Final predictions
_, _, _, y_hat = forward(X, W1, b1, W2, b2)
y_pred = (y_hat >= 0.5).astype(int)
final_acc = np.mean(y_pred == y)

# Plot loss and accuracy
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(losses)
ax[0].set_title('Training Loss (BCE)')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')

ax[1].plot(accs)
ax[1].set_title('Training Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')

plt.show()
print(f"Final accuracy: {final_acc:.3f}")
```

**Visualize Decision Boundary**

```python
def plot_decision_boundary(X, y, W1, b1, W2, b2):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    _, _, _, probs = forward(grid, W1, b1, W2, b2)
    Z = probs.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap='coolwarm', edgecolors='k', s=20)
    plt.title('Neural Network Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

plot_decision_boundary(X, y, W1, b1, W2, b2)
```

What to try:
- Change `hidden_dim` from `2` to `16` and compare fit quality.
- Set `learning_rate` too high (for example `1.0`) and watch training fail.
- Lower epochs and see underfitting.

----

### Wrap-Up

You just used the same optimization idea twice:
- In linear regression, gradient descent adjusted a line.
- In a neural network, gradient descent adjusted many weights through backpropagation.

That shared loop is the core of most modern machine learning training.
