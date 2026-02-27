---
layout: post
title:  Optimization Playground
parent: Minis
nav_order: 6
estimated_time: 2-3 hours
prereqs:
  - Optimization & Gradient Descent theory lesson
  - Numpy basics
outcomes:
  - Run and tune gradient descent experiments
  - Interpret optimization visualizations
difficulty: Beginner
---

## Optimization Playground: Gradient Descent Intuition

If you haven't learned the optimization basics yet, start with the [Gradient Descent Theory lesson](https://astarryknight.github.io/ai-ml/src/theory/grad_desc.html).

### Objective
Build intuition for optimization by training:
1. A linear regression model with simple gradient descent (square trick idea)
2. The same linear model with full-batch gradient descent and a visual loss landscape

### Prerequisites
- Basic Python and NumPy
- Intro to machine learning concepts (features, labels, loss)

### Setup
Everything below is designed to run directly in Google Colab.

### Tasks
- Complete Part 1 and Part 2 with default parameters.
- Re-run with at least two different learning rates and compare outcomes.
- Write down one failure case and one successful configuration.

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

### Part 2: Visualizing the Loss Landscape

Instead of moving to neural networks, let's stay with linear regression and build stronger intuition.

In this part, we optimize the same model:
$$
\hat{y} = wx + b
$$

But now we:
- Use full-batch gradient descent (all points each step)
- Track `(w, b)` over time
- Draw the loss surface contours and show how gradient descent moves downhill

**Batch Gradient Descent Setup**

```python
def mse_loss(x, y, w, b):
    y_hat = w * x + b
    return np.mean((y_hat - y) ** 2)


def gradients(x, y, w, b):
    n = len(x)
    y_hat = w * x + b
    dw = (2 / n) * np.sum((y_hat - y) * x)
    db = (2 / n) * np.sum(y_hat - y)
    return dw, db


def train_batch_gd(x, y, w0=0.0, b0=0.0, learning_rate=0.01, epochs=100):
    w, b = w0, b0
    history = [(w, b, mse_loss(x, y, w, b))]

    for _ in range(epochs):
        dw, db = gradients(x, y, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        history.append((w, b, mse_loss(x, y, w, b)))

    return w, b, history
```

**Run with Different Learning Rates**

```python
# Try different learning rates
rates = [0.001, 0.01, 0.05]
epochs = 120

all_histories = {}
for lr in rates:
    w, b, hist = train_batch_gd(features, labels, w0=0.0, b0=0.0, learning_rate=lr, epochs=epochs)
    all_histories[lr] = (w, b, hist)
    print(f"lr={lr:<6} final w={w:8.3f}, final b={b:8.3f}, final MSE={hist[-1][2]:10.3f}")
```

**Plot Loss vs Epoch**

```python
plt.figure(figsize=(7, 4))
for lr in rates:
    hist = all_histories[lr][2]
    losses = [h[2] for h in hist]
    plt.plot(losses, label=f"lr={lr}")

plt.title("MSE During Training (Batch Gradient Descent)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()
```

**Visualize the Loss Contours + Descent Paths**

```python
# Build a grid of (w, b) values
w_vals = np.linspace(20, 70, 180)
b_vals = np.linspace(50, 180, 180)
W, B = np.meshgrid(w_vals, b_vals)

Z = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = mse_loss(features, labels, W[i, j], B[i, j])

plt.figure(figsize=(8, 6))
contours = plt.contour(W, B, Z, levels=30, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)

# Overlay paths for each learning rate
for lr in rates:
    hist = all_histories[lr][2]
    ws = [h[0] for h in hist]
    bs = [h[1] for h in hist]
    plt.plot(ws, bs, marker='o', markersize=2, linewidth=1.5, label=f"lr={lr}")

plt.title("Loss Landscape (MSE) and Gradient Descent Paths")
plt.xlabel("w (slope)")
plt.ylabel("b (intercept)")
plt.legend()
plt.show()
```

What to try:
- Add a very large learning rate like `0.2` and look for overshooting/divergence.
- Change the start point in `train_batch_gd` (`w0`, `b0`) and compare paths.
- Increase `epochs` and see how quickly each learning rate reaches the minimum region.

----

### Wrap-Up

You just used the same optimization idea twice:
- In linear regression, gradient descent adjusted a line.
- In the loss-landscape view, you watched `(w, b)` physically move downhill on the error surface.

That shared loop is the core of most modern machine learning training.

----

### Validation
- You can explain why different learning rates produce different trajectories.
- Your plots show both convergence and at least one unstable/slow case.
- You can report final RMSE/MSE values and compare settings.

### Extensions
- Add noise to the toy dataset and observe loss landscape changes.
- Try normalizing inputs and compare convergence speed.
- Plot parameter updates (`w`, `b`) versus epoch for one run.

### Deliverable
- A Colab notebook that runs top-to-bottom.
- Two figure outputs: fit/loss plot and contour/path plot.
- A short markdown summary of: best setting, failed setting, and why.
