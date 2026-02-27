---
layout: post
title:  6. Evaluation Basics
parent: Theory
nav_order: 8
estimated_time: 60-90 minutes
prereqs:
  - Intro to Machine Learning
  - Basic Python/NumPy
outcomes:
  - Explain train/validation/test roles
  - Pick basic metrics for regression/classification
  - Identify leakage and common evaluation mistakes
checkpoint: Module 3
---

## Evaluation Basics
Good models are built by iteration, and good iteration requires reliable evaluation.

A practical loop is:
- **Split** data
- **Train** on training data
- **Evaluate** on validation/test data
- **Iterate** with justified changes

----

### Data Splits
- **Training set:** fit model parameters.
- **Validation set:** compare model or hyperparameter choices.
- **Test set:** final unbiased performance estimate.

If data is limited, use cross-validation for more stable estimates.

----

### Common Metrics
**Regression**
- MAE: average absolute error.
- RMSE: penalizes larger errors more strongly.
- R^2: fraction of variance explained.

**Classification**
- Accuracy: overall correct predictions.
- Precision: among predicted positives, how many are true positives.
- Recall: among actual positives, how many were found.
- F1: balance between precision and recall.

Pick metrics based on task costs, not habit.

----

### Data Leakage (High Priority)
Leakage happens when training uses information that would not be available at prediction time.

Common sources:
- Scaling/encoding with full dataset before splitting
- Features derived from target or future information
- Duplicate rows crossing train/test boundaries

Rule: split first, then fit preprocessing on train only.

----

### Error Analysis Checklist
When performance is weak, check:
1. Data quality (missing values, label noise, class imbalance)
2. Feature quality (signal and redundancy)
3. Underfitting/overfitting signs
4. Metric-task mismatch
5. Baseline comparison (is your model actually better?)

----

### Quick Self-Check
1. Why is test data used only at the end?
2. In a fraud problem, when is recall more important than accuracy?
3. Give one leakage example from tabular data workflows.

For answer guidance, see [Checkpoints](../assessments/checkpoints.html).
