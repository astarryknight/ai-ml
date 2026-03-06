---
layout: post
title:  Artificial Neural Network
parent: Minis
nav_order: 3
estimated_time: 2-3 hours
prereqs:
  - Artificial Neural Networks theory lesson
  - Basic Python and NumPy
outcomes:
  - Train a nonlinear PyTorch MLP for binary classification
  - Visualize the dataset, training curves, and final decision boundaries
  - Compare nonlinear and linear models on the same dataset
difficulty: Intermediate
---

## Artificial Neural Network Mini: MLP First, Linear Comparison Second

If you have not done the ANN theory lesson yet, start [here](https://astarryknight.github.io/ai-ml/src/theory/ann.html).

### Objective
Train a small nonlinear neural network in PyTorch, inspect its training metrics, and then compare it to a linear baseline.

### Why this dataset?
We use `sklearn.datasets.make_moons` because it is perfect for visual intuition:
- You can plot every point directly.
- It clearly shows why nonlinear models can outperform linear ones.

### Setup
Run in Google Colab (or local Python):

```python
!pip -q install torch scikit-learn matplotlib
```

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

```

---

## Part 1: Build an Interesting Dataset

This section prepares the data so the network can learn from it cleanly:
- `make_moons(...)` creates a two-class dataset with a curved shape, which is useful because a straight line will struggle to separate it.
- `train_test_split(...)` holds out part of the data for testing, so we can measure whether the model generalizes beyond the examples it trained on.
- `StandardScaler()` rescales each feature so values are centered near `0` with similar spread. Neural networks usually train more smoothly when inputs are on a comparable scale.
- `torch.tensor(...)` converts NumPy arrays into PyTorch tensors, which are the objects the neural network and optimizer expect.

```python
# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Nonlinear, visually rich dataset
X, y = make_moons(n_samples=800, noise=0.22, random_state=42)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

plt.figure(figsize=(6, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.75)
plt.title("Training Data: Two Moons")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
```

Code walkthrough:
- `np.random.seed(42)` and `torch.manual_seed(42)` fix the random number generators so you and another learner can reproduce nearly the same dataset and training run.
- `X` stores the input points and `y` stores the labels `0` or `1`.
- `stratify=y` keeps the class balance similar in both the training set and the test set.
- `y.reshape(-1, 1)` changes labels from a 1D list like `[0, 1, 1, ...]` into a column shape that matches the model output.
- The scatter plot gives a visual answer to: "What pattern is the model trying to learn?"

---

## Part 2: Define the Nonlinear Neural Network (MLP)

We will start with a tiny MLP:
- 2 inputs (`x1`, `x2`)
- 1 hidden layer with 4 neurons + ReLU
- 1 output neuron + sigmoid

Conceptually, the model works in two steps:
- The first layer mixes the two input features into four hidden values.
- The ReLU activation lets the model create nonlinear shapes instead of only straight-line boundaries.
- The second layer combines those hidden values into one score.
- The sigmoid squashes that score into a probability between `0` and `1`.

```python
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out


criterion = nn.BCELoss()
model = TinyMLP()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

Code walkthrough:
- `nn.Linear(2, 4)` means "take 2 input numbers and produce 4 learned combinations of them."
- `torch.relu(...)` replaces negative values with `0`, which gives the network extra flexibility.
- `nn.Linear(4, 1)` reduces the hidden representation down to one output value for binary classification.
- `nn.BCELoss()` is binary cross-entropy loss, which penalizes confident wrong probabilities more heavily than small mistakes.
- `optim.Adam(...)` is the training algorithm that updates the weights a little each step based on the gradients.

---

## Part 3: Train and Track Metrics

We record:
- train/test loss
- train/test accuracy

The training loop repeats the same pattern:
1. Make predictions on the training data.
2. Measure how wrong those predictions are.
3. Backpropagate to compute gradients.
4. Update the weights.
5. Measure loss and accuracy so we can see whether learning is improving over time.

```python
def evaluate(model, X_t, y_t, criterion):
    model.eval()
    with torch.no_grad():
        probs = model(X_t)
        preds = (probs >= 0.5).float()
        loss = criterion(probs, y_t).item()
        acc = preds.eq(y_t).float().mean().item()
    return loss, acc


num_epochs = 250
history = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
}

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    probs = model(X_train_t)
    loss = criterion(probs, y_train_t)
    loss.backward()
    optimizer.step()

    tr_loss, tr_acc = evaluate(model, X_train_t, y_train_t, criterion)
    te_loss, te_acc = evaluate(model, X_test_t, y_test_t, criterion)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["test_loss"].append(te_loss)
    history["test_acc"].append(te_acc)

print(f"Final MLP test accuracy: {history['test_acc'][-1]:.3f}")
```

Code walkthrough:
- `model.train()` turns on training behavior; `model.eval()` switches to evaluation behavior. This matters more in larger models with layers like dropout, but it is still good practice here.
- `optimizer.zero_grad()` clears gradients from the previous step. Without this, PyTorch would add new gradients on top of old ones.
- `loss.backward()` computes how each parameter contributed to the error.
- `optimizer.step()` uses those gradients to update the weights.
- `probs >= 0.5` converts predicted probabilities into class labels.
- `history` stores the metrics from every epoch so the next section can plot the learning curves.

---

## Part 4: Visualization 1 - Training Curves

```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(history["train_loss"], label="Train")
ax[0].plot(history["test_loss"], label="Test")
ax[0].set_title("Loss vs Epoch")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("BCE Loss")
ax[0].legend()

ax[1].plot(history["train_acc"], label="Train")
ax[1].plot(history["test_acc"], label="Test")
ax[1].set_title("Accuracy vs Epoch")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

plt.show()
```

---

## Part 5: Visualization 2 - Final MLP Decision Boundary

```python
# Grid for boundary plots
x_min, x_max = X_train[:, 0].min() - 1.0, X_train[:, 0].max() + 1.0
y_min, y_max = X_train[:, 1].min() - 1.0, X_train[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_t = torch.tensor(grid, dtype=torch.float32)

with torch.no_grad():
    p_mlp = model(grid_t).numpy().reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, p_mlp, levels=25, cmap="coolwarm", alpha=0.35)
plt.contour(xx, yy, p_mlp, levels=[0.5], colors="black", linewidths=2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k", s=25)
plt.title(f"Final MLP Decision Boundary | test acc={history['test_acc'][-1]:.3f}")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
```

---

## Part 6: End Comparison - Linear Model vs MLP

Now use a linear baseline and compare it to the already-trained MLP.

```python
class LinearBinaryNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_binary_model(model, X_train_t, y_train_t, X_test_t, y_test_t, lr=0.01, epochs=250):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history_local = {"test_loss": [], "test_acc": []}

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        probs = model(X_train_t)
        loss = criterion(probs, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            te_probs = model(X_test_t)
            te_preds = (te_probs >= 0.5).float()
            history_local["test_loss"].append(criterion(te_probs, y_test_t).item())
            history_local["test_acc"].append(te_preds.eq(y_test_t).float().mean().item())

    return history_local


def predict_grid(model, xx, yy):
    model.eval()
    points = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        probs = model(torch.tensor(points, dtype=torch.float32)).numpy().reshape(xx.shape)
    return probs


# Train linear baseline
torch.manual_seed(42)
linear_model = LinearBinaryNN()
linear_hist = train_binary_model(
    linear_model,
    X_train_t, y_train_t,
    X_test_t, y_test_t,
    lr=0.01,
    epochs=num_epochs,
)

mlp_final_acc = history["test_acc"][-1]
linear_final_acc = linear_hist["test_acc"][-1]

print(f"MLP final test accuracy:    {mlp_final_acc:.3f}")
print(f"Linear final test accuracy: {linear_final_acc:.3f}")
```

```python
# Final boundary comparison
p_mlp = predict_grid(model, xx, yy)
p_linear = predict_grid(linear_model, xx, yy)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

ax[0].contourf(xx, yy, p_mlp, levels=25, cmap="coolwarm", alpha=0.35)
ax[0].contour(xx, yy, p_mlp, levels=[0.5], colors="black", linewidths=2)
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors='k', s=20)
ax[0].set_title(f"Tiny MLP (2->4->1) | test acc={mlp_final_acc:.3f}")
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")

ax[1].contourf(xx, yy, p_linear, levels=25, cmap="coolwarm", alpha=0.35)
ax[1].contour(xx, yy, p_linear, levels=[0.5], colors="black", linewidths=2)
ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors='k', s=20)
ax[1].set_title(f"Linear Model (2->1) | test acc={linear_final_acc:.3f}")
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")

plt.suptitle("Nonlinear vs Linear: Decision Boundaries")
plt.tight_layout()
plt.show()
```

---

## Tasks
1. Change hidden width in the MLP (`4`, `8`, `16`) and compare boundaries.
2. Increase moon noise (`0.05`, `0.22`, `0.35`) and compare MLP vs linear accuracy gaps.
3. Try MLP learning rates (`0.001`, `0.01`, `0.05`) and describe one unstable run.
4. Explain in 3-4 sentences why the nonlinear model fits this dataset better.

## Validation
- MLP training runs end-to-end and reports test accuracy.
- Dataset plot renders correctly.
- Loss and accuracy curves render correctly.
- Final MLP and side-by-side comparison plots render correctly.

## Deliverable
- Notebook with all cells executed.
- One screenshot of the final MLP decision boundary.
- One screenshot of the final linear vs MLP boundary comparison.
- One short paragraph interpreting the comparison.
