---
layout: post
title:  7. Feature Engineering Basics
parent: Theory
nav_order: 9
estimated_time: 60-90 minutes
prereqs:
  - Intro to Machine Learning
  - Numpy lesson
outcomes:
  - Handle missing values and categorical variables
  - Scale features appropriately
  - Build simple reproducible preprocessing pipelines
checkpoint: Module 3
---

## Feature Engineering Basics
Feature engineering is the bridge between raw data and usable model input.

The goal is not to add complexity. The goal is to make data representation reliable and informative.

----

### 1) Missing Values
Common strategies:
- Drop rows/columns (when impact is small)
- Numeric imputation: mean/median
- Categorical imputation: most frequent or explicit "unknown"

Use training data statistics for imputation.

### 2) Categorical Variables
Common strategies:
- One-hot encoding for nominal categories
- Ordinal encoding only when ordering is meaningful

Watch out for high-cardinality categories and rare labels.

### 3) Feature Scaling
Some models are sensitive to scale (linear models, SVMs, k-NN, neural nets).

Common scalers:
- Standardization (zero mean, unit variance)
- Min-max scaling (fixed range)

Fit scaler on training set only.

### 4) Simple Derived Features
Examples:
- Ratios (price per unit)
- Time deltas (days since last event)
- Domain-specific transformations

Only keep derived features that improve validation performance or interpretability.

----

### Minimal Pipeline Mindset
A reliable pipeline order:
1. Split data
2. Fit preprocessing on train
3. Transform train/validation/test with same fitted preprocessors
4. Train model
5. Evaluate and iterate

----

### Quick Self-Check
1. Why should you not fit a scaler on the full dataset?
2. When is one-hot encoding preferred over ordinal encoding?
3. Name one reasonable way to handle numeric missing values.

For answer guidance, see [Checkpoints](../assessments/checkpoints.html).
