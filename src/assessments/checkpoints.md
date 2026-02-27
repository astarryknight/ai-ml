---
layout: post
title:  Checkpoints
parent: Learning Path
nav_order: 2
estimated_time: 1-2 hours total
outcomes:
  - Validate understanding at each module boundary
  - Catch confusion before moving forward
---

## Module Checkpoints (With Answer Keys)

These are short async self-checks. Try to answer before opening the answer key.

----

### Module 1 Checkpoint: AI/ML Framing
1. In one sentence each, define AI and ML.
2. Give one example where symbolic approaches are useful.
3. Give one example where data-driven ML is useful.

**Answer key:**
- AI is the broader field of building systems that perform tasks associated with intelligence.
- ML is a subset of AI where systems learn patterns from data.
- Symbolic example: rule-based tax form validation.
- ML example: email spam detection from historical labels.

### Module 2 Checkpoint: NumPy Workflow
1. What is broadcasting?
2. Write a boolean mask that selects values greater than 10 in array `a`.
3. Why are vectorized operations preferred over Python loops for arrays?

**Answer key:**
- Broadcasting expands compatible dimensions for element-wise operations.
- `a[a > 10]`
- Vectorized operations are faster and simpler because they use optimized low-level routines.

### Module 3 Checkpoint: Supervised Workflow
1. What is data leakage?
2. Why do we split train/validation/test?
3. Name one regression metric and one classification metric.

**Answer key:**
- Leakage is accidentally letting target/future information enter training features.
- Split enables training, tuning, and unbiased final evaluation.
- Regression: RMSE/MAE. Classification: accuracy/precision/recall/F1.

### Module 4 Checkpoint: Optimization
1. What does learning rate control?
2. What happens when learning rate is too high?
3. What does a flattening loss curve usually indicate?

**Answer key:**
- Step size of each gradient update.
- Overshooting/divergence or unstable training.
- Convergence or near-convergence to a local/global minimum.

### Module 5 Checkpoint: Unsupervised Learning
1. What is clustering used for?
2. Why can interpreting clusters require human judgment?
3. What does dimensionality reduction preserve?

**Answer key:**
- Grouping similar data points without labels.
- Clusters are unlabeled by default and need domain context.
- As much important structure/information as possible in fewer dimensions.

### Module 6 Checkpoint: Neural Networks
1. What is a perceptron?
2. Why are nonlinear activations important?
3. Name one overfitting mitigation strategy.

**Answer key:**
- Weighted sum + bias passed through an activation.
- They allow modeling non-linear boundaries.
- Regularization, dropout, or early stopping.

### Module 7 Checkpoint: NLP
1. Difference between token and type?
2. Why remove stopwords in some tasks?
3. What is lemmatization?

**Answer key:**
- Token is an occurrence; type is a unique token value.
- To reduce noise and focus on informative content words.
- Mapping word forms to root/base form.

### Module 8 Checkpoint: RL
1. Define state, action, reward.
2. What is exploration vs. exploitation?
3. Why use a discount factor?

**Answer key:**
- State: current environment signal; action: agent decision; reward: feedback signal.
- Exploration tries new actions; exploitation picks known high-value actions.
- It balances immediate vs future rewards and stabilizes long-horizon credit assignment.

### Module 9 Checkpoint: Consolidation
1. List two minis you completed and why you chose them.
2. For one mini, describe one iteration you made after evaluating output.
3. Name one limitation in your final result and one next improvement.

**Answer key guidance:**
- Free response; quality should show clear reasoning, evaluation evidence, and realistic next steps.
