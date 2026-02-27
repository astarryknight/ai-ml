---
layout: post
title:  Learning Path
nav_order: 3
estimated_time: 10-12 weeks
prereqs:
  - Basic Python syntax
  - Comfortable running notebooks
outcomes:
  - Follow an end-to-end async pathway
  - Complete core mini artifacts
---

## Learning Path (Async, Self-Paced)

Use this page as the default sequence for the curriculum. Each module has a recommended order, estimated pace, and a checkpoint from the [Checkpoints page](./assessments/checkpoints.html).

----

### Module 1: Orientation + AI/ML Framing
- **Estimated time:** 2-3 hours
- **Lessons:**
  - [Intro](./intro/intro.html)
  - [What is AI/ML?](./intro/whatisai.html)
  - [Local Setup (optional)](./intro/local.html)
- **Goal:** Understand scope, terminology, and tooling options.
- **Checkpoint:** Module 1 on the checkpoints page.

### Module 2: Python/NumPy Data Workflow Basics
- **Estimated time:** 4-6 hours
- **Lessons:**
  - [Numpy](./intro/numpy.html)
- **Goal:** Build and manipulate arrays, masks, and vectorized operations for ML data workflows.
- **Checkpoint:** Module 2.

### Module 3: ML Fundamentals + Evaluation
- **Estimated time:** 5-7 hours
- **Lessons:**
  - [Intro to Machine Learning](./theory/intro_ml.html)
  - [Feature Engineering Basics](./theory/feature_engineering.html)
  - [Evaluation Basics](./theory/evaluation.html)
- **Goal:** Learn data split/train/evaluate/iterate loop with leakage awareness.
- **Checkpoint:** Module 3.

### Module 4: Linear Models + Optimization Intuition
- **Estimated time:** 5-7 hours
- **Lessons:**
  - [Optimization & Gradient Descent](./theory/grad_desc.html)
- **Mini (core):** [Optimization Playground](./minis/opt.html)
- **Goal:** Understand how parameters update and how learning rate affects convergence.
- **Checkpoint:** Module 4.

### Module 5: Unsupervised Learning
- **Estimated time:** 6-8 hours
- **Lessons:**
  - [Intro to Machine Learning](./theory/intro_ml.html) - clustering and dimensionality reduction sections
- **Mini (core):** [Spotify Music Recommendation](./minis/spotify.html)
- **Goal:** Build intuition for clustering workflows and interpretation.
- **Checkpoint:** Module 5.

### Module 6: Intro Neural Networks (Extension)
- **Estimated time:** 6-8 hours
- **Lessons:**
  - [Artificial Neural Networks](./theory/ann.html)
- **Mini (optional core-extension):** [Artificial Neural Network Mini](./minis/ann.html)
- **Goal:** Understand perceptrons, layers, and basic training intuition.
- **Checkpoint:** Module 6.

### Module 7: NLP Foundations
- **Estimated time:** 6-8 hours
- **Lessons:**
  - [Natural Language Processing](./theory/nlp.html)
- **Mini:** [NLP with Disaster Tweets](./minis/nlp.html)
- **Goal:** Build practical text preprocessing and representation workflow.
- **Checkpoint:** Module 7.

### Module 8: RL Foundations
- **Estimated time:** 6-8 hours
- **Lessons:**
  - [Reinforcement Learning](./theory/rl.html)
- **Mini (local recommended):** [Snake with RL](./minis/rl.html)
- **Goal:** Understand state/action/reward and exploration-exploitation tradeoffs.
- **Checkpoint:** Module 8.

### Module 9: Consolidation via Mini Selection
- **Estimated time:** 8-12 hours
- **Goal:** Complete at least two polished minis from different domains (for example Optimization + NLP).
- **Output:** Reproducible notebooks and short markdown summaries.
- **Checkpoint:** Module 9.

----

## Core Mini Set (Mastery Signal)
To complete the async pathway, finish this core set:
- [Optimization Playground](./minis/opt.html)
- [Spotify Music Recommendation](./minis/spotify.html)
- One of: [NLP Mini](./minis/nlp.html), [ANN Mini](./minis/ann.html), or [RL Mini](./minis/rl.html)

Each mini should include a cleaned notebook, output plots/tables, and a short write-up of decisions and results.
