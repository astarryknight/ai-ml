---
layout: post
title:  Code Snippets
parent: Resources
nav_order: 1
estimated_time: 15-30 minutes per snippet
outcomes:
  - Copy a small example into Colab
  - Understand what each line changes
  - Link a snippet to a complete weekly notebook
---

# Code Snippets

Use these small examples when you remember the idea but need a compact code pattern.

## How to use this page

These snippets are short references, not replacements for the guided notebooks. Copy one into Google Colab, run it, and change one value before moving on.

### Week 1: NumPy and a first graph

```python
import numpy as np
import matplotlib.pyplot as plt

hours = np.array([1, 2, 3, 4, 5, 6])
scores = np.array([52, 57, 63, 68, 74, 79])

plt.plot(hours, scores, marker="o")
plt.xlabel("Hours practiced")
plt.ylabel("Score")
plt.title("A first look at a small dataset")
plt.show()
```

Expected output: one line graph with six points. A graph can show a pattern without proving cause.

### Week 1: Shape and masks

```python
data = np.array([[2, 5, 7], [3, 6, 8]])

print(data.shape)
print(data.mean(axis=0))
print(data[data > 5])
```

Expected output: the array shape, one mean per column, and the values greater than 5.

### Week 2: A reproducible train/test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

What changed: the model can now be evaluated on examples it did not train on.

### Week 3: Classification metrics

```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score

print(confusion_matrix(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))
```

What changed: the evaluation reports the types of errors, not only one overall accuracy number.

### Week 4: Preprocessing inside a pipeline

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
```

What changed: preprocessing and training stay together, which reduces accidental leakage.

### Week 5: K-means clustering

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
labels = KMeans(n_clusters=4, random_state=42, n_init="auto").fit_predict(X_scaled)
```

What changed: each row receives a cluster label based on similarity in the scaled feature space.

## Next step

For complete guided notebooks, lessons, and available slides, return to the [weekly guide](../index.html#catch-up-by-week).
