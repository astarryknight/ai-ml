---
layout: post
title:  Evaluation Basics
parent: Core Concepts
nav_order: 3
estimated_time: 60-90 minutes
prereqs:
  - Intro to Machine Learning
  - Basic Python/NumPy
outcomes:
  - Explain train/validation/test roles
  - Pick basic metrics for regression/classification
  - Identify leakage and common evaluation mistakes
---

# Evaluation Basics

Evaluation tells you whether a model is useful on new examples—not merely whether it memorized the examples it saw.

**Suggested week:** 3–4 · **Time:** 45–60 minutes · **Prerequisites:** Intro to Machine Learning

## What you will learn

- Explain the roles of training, validation, and test data.
- Select a basic regression or classification metric.
- Recognize common forms of data leakage.

## Core Path

### Data splits {#data-splits}

- The **training set** is used to fit model parameters.
- The **validation set** is used to compare development choices.
- The **test set** is used once near the end for a less biased estimate of performance.

This [train, validation, and test split](../glossary.html#train-validation-test-split) keeps development decisions from quietly adapting to the final evaluation examples. When data is limited, cross-validation can provide a more stable development estimate.

### Common metrics {#common-metrics}

For regression:

- **MAE** averages the absolute size of errors.
- **RMSE** gives larger errors more influence.
- **R²** compares the model with predicting the target mean.

For classification:

- [**Accuracy**](../glossary.html#accuracy) is the share of predictions that are correct.
- [**Precision**](../glossary.html#precision-recall) asks how many predicted positives are truly positive.
- **Recall** asks how many actual positives the model finds.
- [**F1 score**](../glossary.html#f1-score) balances precision and recall.

Choose a [metric](../glossary.html#metric) from the consequences of mistakes. In a safety screen, missing a real danger may matter more than raising an extra alert.

### Data leakage {#data-leakage}

[Data leakage](../glossary.html#data-leakage) occurs when training uses information that would not be available at prediction time. It can make a weak model look excellent.

Common sources include:

- Scaling or encoding the full dataset before splitting.
- Features derived from the target or future information.
- Duplicate examples crossing training and test sets.

A safe default is: split first, then fit preprocessing on the training data only.

## Try it: choose the metric

For a system that flags urgent support requests, decide whether precision or recall deserves more attention. Write one sentence describing the harm of a false positive and one describing the harm of a false negative.

There is no universal answer—the better metric follows from which mistake is more costly to the user.

## Error-analysis checklist {#error-analysis-checklist}

When performance is weak, check:

1. Data quality: missing values, label noise, and class imbalance.
2. Feature quality: useful signal versus redundant inputs.
3. [Underfitting or overfitting](../glossary.html#overfitting-underfitting).
4. A mismatch between the metric and real task.
5. Whether the model actually improves on a simple baseline.

## Quick self-check

1. Why should test data be used only near the end?
2. When could recall matter more than accuracy?
3. Give one example of leakage in a tabular-data workflow.

For answer guidance, see [Knowledge Checks](../assessments/checkpoints.html).

## Optional Depth

After the core path, explore cross-validation, confidence intervals, calibration, threshold tuning, and metrics designed for imbalanced data. These refine evaluation but do not change its central goal: estimate behavior on relevant new examples.

## Next step

Continue to [Feature Engineering Basics](./feature_engineering.html) to keep preprocessing reproducible and prevent leakage.
