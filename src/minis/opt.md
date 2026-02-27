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

**Code Breakdown**
<!-- - `import numpy as np`: loads NumPy for numeric arrays and math.
- `import matplotlib.pyplot as plt`: loads plotting utilities.
- `import random`: used for stochastic point selection in Part 1. -->
- `features`: input variable (`x`) = number of donations.
- `labels`: target variable (`y`) = funds raised.
- `plt.scatter(features, labels)`: plots raw data so you can visually check if a line fit makes sense.
- If the points roughly align linearly, gradient descent for a line model is a good first approach.

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

**Code Breakdown**
- `rmse(...)`: computes average prediction error magnitude (lower is better).
- In `square_trick(...)`:
- `pred = base_funds + funds_per_dono * num_donos` computes current line prediction for one point.
- `(funds - pred)` is the signed error for that point.
- `funds_per_dono += ...` updates slope (`m`) using error scaled by `x` and learning rate.
- `base_funds += ...` updates intercept (`b`) using error and learning rate.
- In `train_linear(...)`:
- `random.seed(seed)` makes runs reproducible.
- `base_funds` and `funds_per_dono` start random (like random initialization in training).
- Each epoch:
- `preds = ...` predicts all points using current line.
- `errors.append(rmse(...))` stores error history for plotting.
- Random index `i` selects one training point (stochastic gradient descent behavior).
- `square_trick(...)` applies one update step.
- Return values:
- `base_funds`: final intercept.
- `funds_per_dono`: final slope.
- `errors`: learning curve over epochs.

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

**Code Breakdown**
- Hyperparameters:
- `learning_rate`: step size for updates.
- `epochs`: number of update steps.
- `seed`: repeatable randomness.
- `train_linear(...)` returns fitted parameters and error history.
- `preds = ...` computes final fitted line values for plotting.
- Left plot:
- scatter = original data.
- red line = fitted model.
- Right plot:
- RMSE vs epoch = training behavior over time.
- Printed values:
- slope/intercept define the learned line.
- final RMSE summarizes final fit quality.

**How to Read Results**
- Fast drop then plateau in RMSE is expected.
- Very noisy/oscillating RMSE often means learning rate is too high.
- Very slow monotonic decrease often means learning rate is too low.

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

**Code Breakdown**
- `mse_loss(...)`: computes mean squared error for the full dataset.
- `gradients(...)` computes exact full-batch gradients:
- `dw`: how loss changes with slope `w`.
- `db`: how loss changes with intercept `b`.
- Formula intuition:
- if predictions are too high on average, gradients push parameters down.
- if predictions are too low on average, gradients push parameters up.
- `train_batch_gd(...)`:
- starts from `w0`, `b0`.
- stores `(w, b, loss)` at each step in `history`.
- applies update rule:
- `w -= learning_rate * dw`
- `b -= learning_rate * db`
- This is standard gradient descent on a 2-parameter linear model.

**Run with Different Learning Rates**

```python
# Try different learning rates
rates = [0.001, 0.01]
epochs = 120

all_histories = {}
for lr in rates:
    w, b, hist = train_batch_gd(features, labels, w0=0.0, b0=0.0, learning_rate=lr, epochs=epochs)
    all_histories[lr] = (w, b, hist)
    print(f"lr={lr:<6} final w={w:8.3f}, final b={b:8.3f}, final MSE={hist[-1][2]:10.3f}")
```

**Code Breakdown**
- `rates` lets you compare multiple learning rates in one run.
- `all_histories` stores all trajectories for later plotting.
- For each `lr`:
- train model from the same start point (`w0=0.0`, `b0=0.0`).
- collect full trajectory `hist`.
- print final parameters and final MSE.

**What to Observe**
- Which learning rate reaches lower MSE within fixed epochs?
- Which learning rate is unstable or too slow?

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

**Code Breakdown**
- Extract loss sequence from each run: `[h[2] for h in hist]`.
- Plot one curve per learning rate on same axes.
- This gives a direct convergence-speed comparison.

**What to Observe**
- Steeper early decline indicates faster initial learning.
- Flatter curves indicate slower convergence.
- Curves that spike or diverge indicate unstable updates.

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

**Code Breakdown**
- `w_vals`, `b_vals`: parameter grid for contour plot.
- `W, B = np.meshgrid(...)`: creates every `(w, b)` pair on that grid.
- `Z[i, j] = mse_loss(...)`: computes loss at each parameter pair.
- `plt.contour(...)`: draws equal-loss contour lines (a topographic map of loss).
- Overlay section:
- `ws = [h[0] for h in hist]` and `bs = [h[1] for h in hist]` extract each run's path.
- `plt.plot(ws, bs, ...)` shows how gradient descent moves through parameter space.

**How to Interpret the Contour Plot**
- Center/lower contour regions represent lower error.
- A good learning rate traces a smooth path toward low-loss contours.
- Too-large learning rates jump around or overshoot the valley.
- Too-small learning rates move correctly but very slowly.

What to try:
- Add a very large learning rate like `0.02` and look for overshooting/divergence.
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
