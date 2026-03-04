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
  - Visualize decision boundaries and network weight updates during training
  - Compare nonlinear and linear models on the same dataset
difficulty: Intermediate
---

## Artificial Neural Network Mini: MLP First, Linear Comparison Second

If you have not done the ANN theory lesson yet, start [here](https://astarryknight.github.io/ai-ml/src/theory/ann.html).

### Objective
Train a small nonlinear neural network in PyTorch, visualize how it learns, and then compare it to a linear baseline.

### Why this dataset?
We use `sklearn.datasets.make_moons` because it is perfect for visual intuition:
- You can plot every point directly.
- You can watch the decision boundary evolve during training.
- It clearly shows why nonlinear models can outperform linear ones.

### Setup
Run in Google Colab (or local Python):

```python
!pip -q install torch scikit-learn matplotlib ipywidgets networkx
```

```python
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from matplotlib import animation
from IPython.display import HTML
import ipywidgets as widgets
from ipywidgets import interact
import networkx as nx
```

---

## Part 1: Build an Interesting Dataset

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

---

## Part 2: Define the Nonlinear Neural Network (MLP)

We will start with a tiny MLP:
- 2 inputs (`x1`, `x2`)
- 1 hidden layer with 4 neurons + ReLU
- 1 output neuron + sigmoid

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

---

## Part 3: Train and Record Parameter Snapshots

We record:
- train/test loss and accuracy
- full model parameter snapshots each epoch (for animation)

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
    "test_acc": [],
    "states": []
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
    history["states"].append(copy.deepcopy(model.state_dict()))

print(f"Final MLP test accuracy: {history['test_acc'][-1]:.3f}")
```

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

## Part 5: Visualization 2 - Decision Boundary Animation (MLP)

```python
# Grid for boundary plots
x_min, x_max = X_train[:, 0].min() - 1.0, X_train[:, 0].max() + 1.0
y_min, y_max = X_train[:, 1].min() - 1.0, X_train[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_t = torch.tensor(grid, dtype=torch.float32)

# Temporary model used for snapshot playback
tmp_model = TinyMLP()

fig, ax = plt.subplots(figsize=(7, 6))

def animate_boundary(i):
    ax.clear()

    tmp_model.load_state_dict(history["states"][i])
    tmp_model.eval()

    with torch.no_grad():
        p = tmp_model(grid_t).numpy().reshape(xx.shape)

    ax.contourf(xx, yy, p, levels=25, cmap="coolwarm", alpha=0.35)
    ax.contour(xx, yy, p, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=25)

    ax.set_title(
        f"MLP Decision Boundary | Epoch {i+1} | train acc={history['train_acc'][i]:.3f}"
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

ani = animation.FuncAnimation(fig, animate_boundary, frames=num_epochs, interval=90)
plt.close(fig)
HTML(ani.to_jshtml())
```

---

## Part 6: Visualization 3 - Network Graph + Live Weight Changes (MLP)

This graph shows all MLP connections (`2 -> 4 -> 1`).
- Red edge: positive weight
- Blue edge: negative weight
- Thicker edge: larger absolute weight

```python
# Build graph nodes/edges
inputs = ["x1", "x2"]
hiddens = [f"h{i+1}" for i in range(4)]
output = ["y_hat"]

G = nx.DiGraph()
G.add_nodes_from(inputs + hiddens + output)

for x in inputs:
    for h in hiddens:
        G.add_edge(x, h)
for h in hiddens:
    G.add_edge(h, "y_hat")

pos = {
    "x1": (-2, 0.7),
    "x2": (-2, -0.7),
    "h1": (0, 1.2),
    "h2": (0, 0.4),
    "h3": (0, -0.4),
    "h4": (0, -1.2),
    "y_hat": (2, 0),
}


def edge_style(w):
    color = "tab:red" if w >= 0 else "tab:blue"
    width = 0.8 + 4.2 * min(abs(w), 2.0) / 2.0
    return color, width


fig, ax = plt.subplots(figsize=(8, 5))

def animate_nn(i):
    ax.clear()
    state = history["states"][i]

    # fc1: [4, 2], fc2: [1, 4]
    w1 = state["fc1.weight"].cpu().numpy()
    w2 = state["fc2.weight"].cpu().numpy()[0]

    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="#f5f5f5", edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    # Draw input -> hidden edges
    for hi, h in enumerate(hiddens):
        for xi, x in enumerate(inputs):
            w = w1[hi, xi]
            c, lw = edge_style(w)
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(x, h)],
                edge_color=c,
                width=lw,
                arrows=True,
                arrowsize=12,
                ax=ax
            )

    # Draw hidden -> output edges
    for hi, h in enumerate(hiddens):
        w = w2[hi]
        c, lw = edge_style(w)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(h, "y_hat")],
            edge_color=c,
            width=lw,
            arrows=True,
            arrowsize=12,
            ax=ax
        )

    ax.set_title(f"MLP Weights at Epoch {i+1}")
    ax.axis("off")

ani_nn = animation.FuncAnimation(fig, animate_nn, frames=num_epochs, interval=90)
plt.close(fig)
HTML(ani_nn.to_jshtml())
```

---

## Part 7: Interactive Epoch Slider (Optional)

```python
@interact(epoch=widgets.IntSlider(min=0, max=num_epochs-1, step=1, value=num_epochs-1))
def show_epoch(epoch):
    replay_model = TinyMLP()
    replay_model.load_state_dict(history["states"][epoch])
    replay_model.eval()

    with torch.no_grad():
        p = replay_model(grid_t).numpy().reshape(xx.shape)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].contourf(xx, yy, p, levels=25, cmap="coolwarm", alpha=0.35)
    ax[0].contour(xx, yy, p, levels=[0.5], colors='black', linewidths=2)
    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=20)
    ax[0].set_title(f"MLP Boundary @ Epoch {epoch+1}")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")

    ax[1].plot(history["test_acc"], label="MLP test acc")
    ax[1].axvline(epoch, color="black", linestyle="--")
    ax[1].set_title("MLP Test Accuracy Trajectory")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    plt.show()
```

---

## Part 8: End Comparison - Linear Model vs MLP

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
- MLP boundary animation updates as parameters change.
- MLP network-graph animation updates edge style over epochs.
- Final side-by-side linear vs MLP plot renders correctly.

## Deliverable
- Notebook with all cells executed.
- One screenshot/GIF of the MLP boundary animation.
- One screenshot of the final linear vs MLP boundary comparison.
- One short paragraph interpreting the comparison.
