---
layout: post
title: Handwritten Digit Classifier
parent: Practice
nav_order: 3
estimated_time: 90-120 minutes
prereqs:
  - Artificial Neural Networks theory lesson
  - Basic Python and NumPy
outcomes:
  - Train a small neural network for multiclass image classification
  - Compare it with a linear baseline
  - Use a confusion matrix and misclassified images for error analysis
difficulty: Intermediate
---

# Handwritten Digit Classifier

Train a small neural network to recognize handwritten digits, compare it with a linear baseline, and inspect the images it gets wrong.

For the guided Week 6 version, [open the notebook in Colab](https://colab.research.google.com/github/astarryknight/ai-ml/blob/main/notebooks/week-06-digit-classifier.ipynb) or [download it](../../notebooks/week-06-digit-classifier.ipynb). This page is the extended walkthrough.

If you have not completed the concept lesson, start with [Artificial Neural Networks](../theory/ann.html).

## Goal

Use the same workflow you already know:

1. Prepare the data.
2. Build a baseline.
3. Train the neural network.
4. Evaluate both models.
5. Inspect errors.

The neural-network code is deliberately short. The important work is understanding the data and interpreting the result.

## Why this dataset?

The scikit-learn digits dataset is a practical bridge into image classification:

- It is included with scikit-learn, so there is no separate download.
- Every example is a visible 8×8 image.
- Each image becomes 64 numeric pixel features.
- Ten output classes make the task realistic without requiring a large computer.

## Before you start

Run this activity in Google Colab or local Python. Colab already includes the required libraries in most sessions.

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
```

---

## Phase 1: Prepare and inspect the images

Load the data and look at a few examples before training anything.

```python
digits = load_digits()

images = digits.images
X = digits.data / 16.0
y = digits.target

print("Image array:", images.shape)
print("Feature matrix:", X.shape)
print("Labels:", y.shape)
```

Code walkthrough:

- `images` has shape `(1797, 8, 8)`, so it keeps each image's rows and columns.
- `X` has shape `(1797, 64)`, so each image is flattened into one row of model inputs.
- Pixel values originally range from `0` to `16`. Dividing by `16.0` puts them between `0` and `1`.
- `y` contains the correct digit for each image.

### View the data

```python
fig, axes = plt.subplots(2, 6, figsize=(10, 4))

for ax, image, label in zip(axes.ravel(), images[:12], y[:12]):
    ax.imshow(image, cmap="gray_r")
    ax.set_title(f"Label: {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()
```

Before continuing, answer: which pairs of digits might be difficult to distinguish?

### Split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("Training examples:", len(X_train))
print("Test examples:", len(X_test))
```

`stratify=y` keeps the proportion of each digit similar in the training and test sets.

---

## Phase 2: Build a linear baseline

A baseline tells us whether the neural network actually improves the result.

```python
baseline = LogisticRegression(max_iter=2000, random_state=42)
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print(f"Linear baseline accuracy: {baseline_accuracy:.3f}")
```

Logistic regression creates a linear score for each class. It is a strong baseline for this small dataset, so the neural network should earn its added complexity.

---

## Phase 3: Train the neural network

This model has one hidden layer with 32 units:

```text
64 pixel inputs → 32 ReLU hidden units → 10 digit outputs
```

```python
mlp = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.003,
    max_iter=500,
    n_iter_no_change=20,
    random_state=42,
)

mlp.fit(X_train, y_train)

mlp_predictions = mlp.predict(X_test)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)

print(f"Neural-network accuracy: {mlp_accuracy:.3f}")
print(f"Training epochs used: {mlp.n_iter_}")
```

Code walkthrough:

- `hidden_layer_sizes=(32,)` creates one hidden layer containing 32 units.
- `activation="relu"` adds nonlinearity inside that hidden layer.
- `solver="adam"` updates the weights using gradients from backpropagation.
- `learning_rate_init=0.003` controls the initial update size.
- `n_iter_no_change=20` stops training after the loss has stopped improving for several epochs.
- `random_state=42` makes the split and initial model behavior reproducible.

The library performs the matrix operations and backpropagation. Your job is to choose a reasonable experiment and evaluate it honestly.

---

## Phase 4: Inspect learning

`MLPClassifier` stores the loss after each training epoch.

```python
plt.figure(figsize=(7, 4))
plt.plot(mlp.loss_curve_)
plt.title("Neural-network training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
```

Look for the general trend rather than expecting a perfectly smooth curve. The test set remains untouched until evaluation.

---

## Phase 5: Compare the models

```python
print(f"Linear baseline: {baseline_accuracy:.3f}")
print(f"Neural network:  {mlp_accuracy:.3f}")
print(f"Difference:      {mlp_accuracy - baseline_accuracy:+.3f}")
```

Do not assume the neural network must win. On a small dataset, a simpler model may already perform very well. The comparison is useful either way.

Write down:

1. Which model performed better on the test set?
2. Was the difference large enough to matter?
3. What extra cost or complexity does the neural network introduce?

---

## Phase 6: Use a confusion matrix

```python
ConfusionMatrixDisplay.from_predictions(
    y_test,
    mlp_predictions,
    cmap="Blues",
    colorbar=False,
)
plt.title("Neural-network confusion matrix")
plt.show()
```

Rows are the correct digits and columns are the predicted digits. Values away from the diagonal are mistakes.

Find the most common confusion:

```python
mistake_counts = np.zeros((10, 10), dtype=int)

for actual, predicted in zip(y_test, mlp_predictions):
    if actual != predicted:
        mistake_counts[actual, predicted] += 1

actual_digit, predicted_digit = np.unravel_index(
    np.argmax(mistake_counts),
    mistake_counts.shape,
)

print(
    f"Most common confusion: {actual_digit} predicted as {predicted_digit} "
    f"({mistake_counts[actual_digit, predicted_digit]} times)"
)
```

---

## Phase 7: Look at misclassified images

Metrics summarize errors. Images help explain them.

```python
wrong = np.flatnonzero(mlp_predictions != y_test)
count = min(12, len(wrong))

fig, axes = plt.subplots(2, 6, figsize=(11, 4))

for ax in axes.ravel():
    ax.axis("off")

for ax, test_index in zip(axes.ravel(), wrong[:count]):
    ax.imshow(X_test[test_index].reshape(8, 8), cmap="gray_r")
    ax.set_title(
        f"True: {y_test[test_index]}\nPred: {mlp_predictions[test_index]}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()
```

Choose two mistakes and describe what might have made them difficult. Avoid claiming that you know exactly what an individual hidden unit learned unless you test that claim.

---

## Phase 8: Extend the experiment

Change one thing at a time and record the result.

1. Try hidden-layer sizes `(16,)`, `(32,)`, and `(64,)`.
2. Try two hidden layers with `(32, 16)`.
3. Try regularization values `alpha=0.0001`, `0.01`, and `0.1`.
4. Turn on `early_stopping=True` and compare the number of epochs and test accuracy.
5. Inspect whether the same digit pairs remain difficult.

Use a small results table:

| Hidden layers | Alpha | Epochs | Test accuracy | Most common confusion |
| --- | ---: | ---: | ---: | --- |
| `(32,)` | `0.0001` |  |  |  |

## Validate

- The sample digit grid renders.
- The baseline and neural network both report test accuracy.
- The training-loss plot renders.
- The confusion matrix renders.
- Misclassified images show both the correct and predicted labels.
- Your extension changes only one main setting at a time.

## Finish line

- One executed notebook containing the full workflow.
- The baseline and neural-network test accuracies.
- One confusion matrix.
- At least two inspected mistakes.
- A short conclusion explaining whether the neural network was worth the added complexity.

## Optional PyTorch connection

The scikit-learn model hides the training loop, but its architecture maps directly to a small PyTorch model:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)
```

The same inputs, hidden layer, activations, outputs, loss, backpropagation, and evaluation ideas still apply. Use PyTorch when you need more control over the architecture or training process.
