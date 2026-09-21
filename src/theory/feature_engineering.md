---
layout: post
title:  Feature Engineering Basics
parent: Core Concepts
nav_order: 4
estimated_time: 60-90 minutes
prereqs:
  - Intro to Machine Learning
  - Numpy lesson
outcomes:
  - Handle missing values and categorical variables
  - Scale features appropriately
  - Build simple reproducible preprocessing pipelines
---

# Feature Engineering Basics

Feature engineering turns raw data into a reliable representation a model can use.

**Suggested week:** 4 · **Time:** 45–60 minutes · **Prerequisites:** Intro to Machine Learning and Evaluation Basics

## What you will learn

- Handle missing and categorical values without leaking information.
- Explain when feature scaling matters.
- Keep preprocessing and modeling in one reproducible pipeline.

## Core Path

[Preprocessing](../glossary.html#preprocessing) should make data more consistent and informative—not merely more complicated.

### Missing values

Common strategies include dropping a small number of affected rows, filling numeric values with a median, or filling categories with the most frequent value or an explicit `unknown` label. Learn any fill value from the training set, not the complete dataset.

### Categorical variables

Use one-hot encoding when categories have no meaningful order. Use ordinal encoding only when the ordering itself is real. Watch for very high-cardinality columns such as unique IDs; they may not generalize.

### Feature scaling {#feature-scaling}

[Feature scaling](../glossary.html#feature-scaling) puts numeric inputs on comparable ranges. It is important for distance-based methods such as k-nearest neighbors and K-means, and often helps linear models and neural networks.

- **Standardization** centers a feature and scales its spread.
- **Min-max scaling** maps values to a fixed range.

Fit the scaler on training data only, then reuse that fitted scaler on validation, test, and future inputs.

### Simple derived features

Useful derived features express relevant domain information, such as price per unit or days since a prior event. Keep one when it improves validation performance or makes the model easier to interpret.

### Minimal pipeline mindset {#minimal-pipeline-mindset}

A [pipeline](../glossary.html#pipeline) keeps the order reproducible:

1. Split the data.
2. Fit preprocessing on training data.
3. Apply the fitted preprocessing to each split.
4. Train the model.
5. Evaluate and change one thing at a time.

## Try it: spot the unsafe step

A student fills missing values and scales the complete dataset, then creates the train/test split. What should change?

<details>
<summary>Check your answer</summary>

Split first. Learn the fill values and scaling parameters from the training set, then apply those fitted transformations to the test set.

</details>

## Quick self-check

1. Why should a scaler not be fitted on the full dataset?
2. When is one-hot encoding preferable to ordinal encoding?
3. Name one reasonable way to handle a missing numeric value.

For answer guidance, see [Knowledge Checks](../assessments/checkpoints.html).

## Optional Depth

Later, explore feature selection, target encoding with careful validation, transformations for skewed values, and domain-specific representations. Add these only when evaluation shows a benefit.

## Next step

Use the [Week 4 data-pipeline notebook](https://colab.research.google.com/github/astarryknight/ai-ml/blob/main/notebooks/week-04-data-pipeline.ipynb) to apply these steps in code.
