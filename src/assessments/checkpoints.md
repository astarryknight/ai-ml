---
layout: post
title:  Knowledge Checks
parent: Practice
nav_order: 1
estimated_time: 30-45 minutes total
outcomes:
  - Validate understanding at each core boundary
  - Catch confusion before moving forward
  - Review the main ideas at your own pace
---

# Knowledge Checks

Use these short questions to find gaps before moving on. They are practice, not graded assignments.

## How to use these checks

These are short self-checks, not graded assignments. Try to answer each question before opening the answer key. You can use the checks after a lesson or return to them whenever you need a refresher.

| Check | Topic |
| --- | --- |
| 1 | Foundations and NumPy |
| 2 | Regression and optimization |
| 3 | Classification and metrics |
| 4 | Data workflow and generalization |
| 5 | Unsupervised learning and recommendations |

### Check 1: Foundations and NumPy

1. Define AI and ML in one sentence each.
2. Give one example where explicit rules are useful.
3. What does the shape `(2, 3)` tell you about a NumPy array?

<details>
<summary>Answer key</summary>

- AI is the broad field of building systems that perform tasks associated with intelligent behavior.
- ML is a subset of AI where systems learn patterns from data.
- A `(2, 3)` array has 2 rows and 3 columns.

</details>

### Check 2: Regression and optimization

1. What is the difference between a feature and a target?
2. What does a loss value measure?
3. What does the learning rate control in gradient descent?

<details>
<summary>Answer key</summary>

- A feature is an input used for prediction. The target is the value the model tries to predict.
- Loss measures how far predictions are from the desired targets according to a chosen function.
- The learning rate controls the size of each parameter update.

</details>

### Check 3: Classification and metrics

1. What is the difference between regression and classification?
2. Identify one false positive and one false negative in a confusion matrix.
3. When might recall matter more than accuracy?

<details>
<summary>Answer key</summary>

- Regression predicts a numeric value. Classification predicts a category.
- A false positive predicts the positive class when the truth is negative. A false negative misses a positive example.
- Recall matters when missing a positive case is especially costly, such as a safety or medical screening task.

</details>

### Check 4: Data workflow and generalization

1. Why do we split train, validation, and test data?
2. What is data leakage?
3. Give one example of a feature-engineering step.

<details>
<summary>Answer key</summary>

- The split separates fitting, tuning, and final evaluation so the final score better reflects performance on new data.
- Leakage occurs when information that would not be available at prediction time enters the training features or process.
- Examples include scaling numeric features, encoding categories, or filling missing values inside a reproducible pipeline.

</details>

### Check 5: Unsupervised learning and recommendations

1. What is clustering used for?
2. Why can interpreting clusters require human judgment?
3. Why should features often be scaled before K-means?

<details>
<summary>Answer key</summary>

- Clustering groups similar data points without requiring labels.
- A cluster is a similarity group, not automatically a meaningful real-world category.
- K-means uses distances, so features with larger numeric scales can dominate the grouping.

</details>

## Additional practice

### Neural networks

1. What does a layer do?
2. Why are nonlinear activation functions useful?
3. What is the validation set used for?

### Planning your own project

Draw the flow from the user problem to data or retrieval, model or agent, evaluation, and final user-facing output. Use [Build a Project](../capstone.html) to turn the sketch into a one-page proposal.
