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
  - Train a PyTorch linear model as a single-neuron neural network
  - Visualize decision boundaries and model confidence during training
  - Visualize network structure and weight updates over epochs
difficulty: Intermediate
---

## Artificial Neural Network Mini: Dynamic Linear Classifier in PyTorch

If you have not done the ANN theory lesson yet, start [here](https://astarryknight.github.io/ai-ml/src/theory/ann.html).

### Objective
Build and train a neural network with PyTorch (a linear layer + sigmoid), and make training visual.

### Why this dataset?
Instead of bank churn, we will use a 2D moon-shaped dataset generated with `sklearn.datasets.make_moons`.

Why this is better for learning:
- You can plot every point directly.
- You can watch the decision boundary move while training.
- You can see where a linear neural network succeeds and fails.

### Setup
Run in Google Colab (or local Python environment):

```python
!pip -q install torch scikit-learn matplotlib ipywidgets networkx
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

from matplotlib import animation
from IPython.display import HTML
import networkx as nx
```

---

## Part 1: Build an Interesting Dataset

```python
# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Nonlinear, visually interesting dataset
X, y = make_moons(n_samples=800, noise=0.22, random_state=42)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features (important for stable optimization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to torch tensors
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

### Checkpoint
You should see two interleaving crescent shapes. This is intentionally hard for a linear classifier, which makes the visuals more informative.

---

## Part 2: Define the Neural Network (Linear Model)

A linear classifier in PyTorch is still a neural network:
- 2 input neurons (`x1`, `x2`)
- 1 output neuron (logit)
- Sigmoid activation for binary probability

```python
class LinearBinaryNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # 2 inputs -> 1 output

    def forward(self, x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return probs

model = LinearBinaryNN()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.08)
```

---

## Part 3: Train and Record Weight History

We will store:
- loss and accuracy per epoch
- full parameter snapshots (`w1`, `w2`, `b`) for animation

```python
def evaluate(model, X_t, y_t):
    model.eval()
    with torch.no_grad():
        probs = model(X_t)
        preds = (probs >= 0.5).float()
        acc = (preds.eq(y_t).float().mean()).item()
        loss = criterion(probs, y_t).item()
    return loss, acc

num_epochs = 250
history = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": [],
    "w1": [],
    "w2": [],
    "b": []
}

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    probs = model(X_train_t)
    loss = criterion(probs, y_train_t)
    loss.backward()
    optimizer.step()

    # Metrics
    tr_loss, tr_acc = evaluate(model, X_train_t, y_train_t)
    te_loss, te_acc = evaluate(model, X_test_t, y_test_t)

    # Save history
    w = model.linear.weight.detach().numpy().flatten()
    b = model.linear.bias.detach().item()

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["test_loss"].append(te_loss)
    history["test_acc"].append(te_acc)
    history["w1"].append(w[0])
    history["w2"].append(w[1])
    history["b"].append(b)

print(f"Final test accuracy: {history['test_acc'][-1]:.3f}")
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

## Part 5: Visualization 2 - Decision Boundary Animation

This shows how changing weights moves the classifier boundary over time.

```python
# Mesh grid for boundary plotting
x_min, x_max = X_train[:, 0].min() - 1.0, X_train[:, 0].max() + 1.0
y_min, y_max = X_train[:, 1].min() - 1.0, X_train[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
grid = np.c_[xx.ravel(), yy.ravel()]

def predict_grid_from_params(grid_np, w1, w2, b):
    z = w1 * grid_np[:, 0] + w2 * grid_np[:, 1] + b
    p = 1 / (1 + np.exp(-z))
    return p.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 6))

def animate_boundary(i):
    ax.clear()
    p = predict_grid_from_params(grid, history["w1"][i], history["w2"][i], history["b"][i])

    ax.contourf(xx, yy, p, levels=25, cmap="coolwarm", alpha=0.35)
    ax.contour(xx, yy, p, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=25)

    ax.set_title(
        f"Epoch {i+1} | train acc={history['train_acc'][i]:.3f} | "
        f"w=[{history['w1'][i]:.2f}, {history['w2'][i]:.2f}] b={history['b'][i]:.2f}"
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

ani = animation.FuncAnimation(fig, animate_boundary, frames=num_epochs, interval=90)
plt.close(fig)
HTML(ani.to_jshtml())
```

---

## Part 6: Visualization 3 - Neural Network Graph + Live Weights

Even though this is a simple network, we can still draw it and animate edge thickness/color by weight value.

```python
# Graph structure: x1, x2 -> y_hat
G = nx.DiGraph()
G.add_nodes_from(["x1", "x2", "y_hat"])
G.add_edges_from([("x1", "y_hat"), ("x2", "y_hat")])

pos = {
    "x1": (-1, 0.5),
    "x2": (-1, -0.5),
    "y_hat": (1, 0.0)
}

fig, ax = plt.subplots(figsize=(6, 4))

def edge_style(w):
    color = "tab:red" if w >= 0 else "tab:blue"
    width = 1 + 5 * min(abs(w), 2.0) / 2.0
    return color, width


def animate_nn(i):
    ax.clear()
    w1 = history["w1"][i]
    w2 = history["w2"][i]
    b = history["b"][i]

    nx.draw_networkx_nodes(G, pos, node_size=2200, node_color="#f5f5f5", edgecolors="black", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, ax=ax)

    c1, lw1 = edge_style(w1)
    c2, lw2 = edge_style(w2)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=[("x1", "y_hat")],
        edge_color=c1,
        width=lw1,
        arrows=True,
        arrowsize=20,
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[("x2", "y_hat")],
        edge_color=c2,
        width=lw2,
        arrows=True,
        arrowsize=20,
        ax=ax
    )

    ax.text(0, 0.75, f"w1={w1:.3f}", ha="center")
    ax.text(0, -0.75, f"w2={w2:.3f}", ha="center")
    ax.text(1, -0.35, f"b={b:.3f}", ha="center")
    ax.set_title(f"Neural Network Weights at Epoch {i+1}")
    ax.axis("off")

ani_nn = animation.FuncAnimation(fig, animate_nn, frames=num_epochs, interval=90)
plt.close(fig)
HTML(ani_nn.to_jshtml())
```

What to look for:
- Red edges = positive influence, blue edges = negative influence.
- Thicker edge = larger absolute weight.
- As weights shift, the decision boundary in Part 5 rotates/translates.

---

## Part 7: Interactive Epoch Slider (Optional)

If you want manual control instead of autoplay animation:

```python
import ipywidgets as widgets
from ipywidgets import interact

@interact(epoch=widgets.IntSlider(min=0, max=num_epochs-1, step=1, value=num_epochs-1))
def show_epoch(epoch):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: decision boundary
    p = predict_grid_from_params(grid, history["w1"][epoch], history["w2"][epoch], history["b"][epoch])
    ax[0].contourf(xx, yy, p, levels=25, cmap="coolwarm", alpha=0.35)
    ax[0].contour(xx, yy, p, levels=[0.5], colors='black', linewidths=2)
    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=20)
    ax[0].set_title(f"Decision Boundary @ Epoch {epoch+1}")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")

    # Right: parameter history with marker
    ax[1].plot(history["w1"], label="w1")
    ax[1].plot(history["w2"], label="w2")
    ax[1].plot(history["b"], label="b")
    ax[1].axvline(epoch, color="black", linestyle="--")
    ax[1].set_title("Parameter Trajectories")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Value")
    ax[1].legend()

    plt.show()
```

---

## Part 8: Linear vs Nonlinear (Side-by-Side)

Now compare:
- Linear model: `nn.Linear(2, 1)` + sigmoid
- Nonlinear model: `2 -> 8 -> 1` with `ReLU`

```python
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, X_train_t, y_train_t, X_test_t, y_test_t, lr=0.05, epochs=300):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss, test_loss, train_acc, test_acc = [], [], [], []
    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        probs = model(X_train_t)
        loss = criterion(probs, y_train_t)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            tr_probs = model(X_train_t)
            te_probs = model(X_test_t)

            tr_preds = (tr_probs >= 0.5).float()
            te_preds = (te_probs >= 0.5).float()

            train_loss.append(criterion(tr_probs, y_train_t).item())
            test_loss.append(criterion(te_probs, y_test_t).item())
            train_acc.append(tr_preds.eq(y_train_t).float().mean().item())
            test_acc.append(te_preds.eq(y_test_t).float().mean().item())

    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "test_acc": test_acc
    }


def predict_grid(model, xx, yy):
    model.eval()
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        probs = model(torch.tensor(grid_points, dtype=torch.float32)).numpy().reshape(xx.shape)
    return probs


# Train both models from fresh initialization
torch.manual_seed(42)
linear_model = LinearBinaryNN()
linear_hist = train_model(linear_model, X_train_t, y_train_t, X_test_t, y_test_t, lr=0.05, epochs=300)

torch.manual_seed(42)
mlp_model = TinyMLP()
mlp_hist = train_model(mlp_model, X_train_t, y_train_t, X_test_t, y_test_t, lr=0.01, epochs=300)

print(f"Linear final test acc: {linear_hist['test_acc'][-1]:.3f}")
print(f"MLP final test acc:    {mlp_hist['test_acc'][-1]:.3f}")
```

```python
# Side-by-side boundary comparison
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

p_linear = predict_grid(linear_model, xx, yy)
p_mlp = predict_grid(mlp_model, xx, yy)

ax[0].contourf(xx, yy, p_linear, levels=25, cmap="coolwarm", alpha=0.35)
ax[0].contour(xx, yy, p_linear, levels=[0.5], colors="black", linewidths=2)
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k", s=20)
ax[0].set_title(f"Linear Model | test acc={linear_hist['test_acc'][-1]:.3f}")
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")

ax[1].contourf(xx, yy, p_mlp, levels=25, cmap="coolwarm", alpha=0.35)
ax[1].contour(xx, yy, p_mlp, levels=[0.5], colors="black", linewidths=2)
ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="k", s=20)
ax[1].set_title(f"Tiny MLP (2->8->1) | test acc={mlp_hist['test_acc'][-1]:.3f}")
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")

plt.suptitle("Linear vs Nonlinear Decision Boundaries")
plt.tight_layout()
plt.show()
```

```python
# Optional: compare learning curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(linear_hist["test_loss"], label="Linear")
ax[0].plot(mlp_hist["test_loss"], label="Tiny MLP")
ax[0].set_title("Test Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("BCE Loss")
ax[0].legend()

ax[1].plot(linear_hist["test_acc"], label="Linear")
ax[1].plot(mlp_hist["test_acc"], label="Tiny MLP")
ax[1].set_title("Test Accuracy")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

plt.show()
```

Expected takeaway:
- Linear model learns one straight split.
- Tiny MLP bends the boundary to match the moon shapes better.

---

## Tasks
1. Change `noise` in `make_moons` to `0.05`, `0.22`, and `0.35`. Compare final accuracy and boundary shape.
2. Try learning rates `0.01`, `0.08`, `0.2`. Explain one unstable run.
3. Increase epochs to `600` and inspect whether weights stabilize.
4. In Part 8, change hidden size from `8` to `3` and `32`. Compare boundary smoothness and generalization.

## Validation
- Training runs end-to-end with PyTorch.
- Final test accuracy is printed.
- Boundary animation updates as parameters change.
- Network graph animation updates edge styles as weights change.
- Side-by-side linear vs MLP boundary plot renders correctly.

## Deliverable
- Notebook with all cells executed.
- One screenshot/GIF of decision-boundary animation.
- One short paragraph explaining how `w1`, `w2`, and `b` changed model behavior.
