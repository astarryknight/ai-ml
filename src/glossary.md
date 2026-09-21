---
layout: post
title: ML Glossary
parent: Resources
nav_order: 2
---

# ML Glossary

Machine-learning language can sound more intimidating than the underlying ideas. Use this page for a quick explanation, then follow **Learn more** when you want the full lesson.

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [F](#f) · [G](#g) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [R](#r) · [S](#s) · [T](#t) · [U](#u)

## A {#a}

### Accuracy {#accuracy}

The share of predictions a classifier gets right. It can be misleading when one class is much more common than another. **Learn more:** [Evaluation Basics](./theory/evaluation.html#common-metrics).

### Activation function {#activation-function}

A rule applied inside a neural network that helps it represent nonlinear patterns. Without nonlinear activations, stacking layers would still behave like one linear transformation. **Learn more:** [Artificial Neural Networks](./theory/ann.html#activation-functions).

### Agent {#agent}

The decision-maker in reinforcement learning. The agent observes an environment, chooses actions, and learns from rewards. **Learn more:** [Reinforcement Learning](./theory/rl.html#agents-environments).

### Algorithm {#algorithm}

A defined procedure for completing a task. A training algorithm is the set of steps used to create or update a model. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#the-workflow-loop).

### Artificial intelligence (AI) {#artificial-intelligence}

The broad field of building systems that perform tasks associated with intelligent behavior. Machine learning is one approach within AI. **Learn more:** [What is AI/ML?](./intro/whatisai.html).

## B {#b}

### Backpropagation {#backpropagation}

The procedure neural networks use to work backward from an error and determine how each parameter contributed to it. **Learn more:** [Artificial Neural Networks](./theory/ann.html#backpropagation).

### Baseline {#baseline}

A simple result used as a reference point. A more complicated model should solve a real problem better than the baseline before its extra complexity is justified. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#the-workflow-loop).

## C {#c}

### Classification {#classification}

Predicting a category, such as spam or not spam. **Learn more:** [What is AI/ML?](./intro/whatisai.html#three-common-ml-problem-types).

### Clustering {#clustering}

Grouping examples by similarity when the desired groups are not already labeled. The groups still require human interpretation. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#clustering).

### Corpus and token {#corpus-token}

A **corpus** is a collection of text used for analysis. A **token** is one unit within that text, such as a word or punctuation mark. **Learn more:** [Natural Language Processing](./theory/nlp.html#corpora-and-tokens).

## D {#d}

### Data leakage {#data-leakage}

Using information during training that would not truly be available when making a future prediction. Leakage can make evaluation look unrealistically good. **Learn more:** [Evaluation Basics](./theory/evaluation.html#data-leakage).

### Dataset and sample {#dataset-sample}

A **dataset** is a collection of examples. One example or row is often called a **sample**. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#features-targets-and-labels).

## F {#f}

### F1 score {#f1-score}

A classification metric that balances precision and recall in one number. **Learn more:** [Evaluation Basics](./theory/evaluation.html#common-metrics).

### Feature {#feature}

An input value a model can use to make a prediction, such as a song's tempo or a house's size. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#features-targets-and-labels).

### Feature scaling {#feature-scaling}

Transforming numeric features so their ranges are comparable. This matters for methods that use distances or gradient-based optimization. **Learn more:** [Feature Engineering Basics](./theory/feature_engineering.html#feature-scaling).

## G {#g}

### Gradient descent {#gradient-descent}

A repeated process that adjusts model parameters in the direction that reduces loss. Picture taking small downhill steps on an error landscape. **Learn more:** [Optimization & Gradient Descent](./theory/grad_desc.html#optimization-gradient-descent).

## L {#l}

### Learning rate {#learning-rate}

The size of each update during gradient descent. Steps that are too large can overshoot; steps that are too small can make learning slow. **Learn more:** [Optimization & Gradient Descent](./theory/grad_desc.html#troubleshooting-gradient-descent).

### Loss and error {#loss-error}

A number describing how far predictions are from desired answers. Training tries to reduce a chosen loss function. **Learn more:** [Optimization & Gradient Descent](./theory/grad_desc.html#optimization-gradient-descent).

## M {#m}

### Machine learning (ML) {#machine-learning}

An approach in which systems learn useful patterns from examples rather than relying only on hand-written rules. **Learn more:** [What is AI/ML?](./intro/whatisai.html).

### Metric {#metric}

A measurement used to judge model performance. The right metric depends on the consequences of different mistakes. **Learn more:** [Evaluation Basics](./theory/evaluation.html#common-metrics).

### Model {#model}

A learned representation that turns inputs into predictions or other useful outputs. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#the-workflow-loop).

## N {#n}

### Neural network {#neural-network}

A model built from connected layers that transform inputs. Training adjusts the connections so the network produces more useful outputs. **Learn more:** [Artificial Neural Networks](./theory/ann.html).

## O {#o}

### Optimization {#optimization}

The process of finding parameter values that make an objective, such as prediction error, better. **Learn more:** [Optimization & Gradient Descent](./theory/grad_desc.html#optimization-gradient-descent).

### Overfitting and underfitting {#overfitting-underfitting}

**Overfitting** means learning the training examples too specifically and struggling on new data. **Underfitting** means the model has not captured enough of the pattern. **Learn more:** [Evaluation Basics](./theory/evaluation.html#error-analysis-checklist).

## P {#p}

### Pipeline {#pipeline}

A reproducible sequence that applies preprocessing and modeling steps in the correct order. **Learn more:** [Feature Engineering Basics](./theory/feature_engineering.html#minimal-pipeline-mindset).

### Precision and recall {#precision-recall}

**Precision** asks how many predicted positives were correct. **Recall** asks how many actual positives were found. **Learn more:** [Evaluation Basics](./theory/evaluation.html#common-metrics).

### Prediction and inference {#prediction-inference}

Using a trained model on an input to produce an output. This use stage is often called inference. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#the-workflow-loop).

### Preprocessing {#preprocessing}

Steps that turn raw data into inputs a model can reliably use, such as filling missing values or encoding categories. **Learn more:** [Feature Engineering Basics](./theory/feature_engineering.html).

## R {#r}

### Regression {#regression}

Predicting a numeric value, such as a price or delivery time. **Learn more:** [What is AI/ML?](./intro/whatisai.html#three-common-ml-problem-types).

### Reinforcement learning {#reinforcement-learning}

A learning setup in which an agent chooses actions and receives rewards from an environment. **Learn more:** [Reinforcement Learning](./theory/rl.html).

### Reward, policy, and value {#reward-policy-value}

A **reward** is feedback after an action. A **policy** describes how an agent chooses actions. A **value** estimates how useful a state or action will be over time. **Learn more:** [Reinforcement Learning](./theory/rl.html#policy-and-value-functions).

## S {#s}

### Supervised learning {#supervised-learning}

Learning from examples that include the desired answer or label. Regression and classification are common supervised tasks. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#supervised-learning).

## T {#t}

### Target and label {#target-label}

The answer a supervised model is trained to predict. “Target” is common for any output; “label” is especially common for categories. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#features-targets-and-labels).

### Training {#training}

The process of using examples to adjust a model. Training is separate from evaluating the finished model on unseen data. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#the-workflow-loop).

### Train, validation, and test split {#train-validation-test-split}

Three data roles: fit on training data, make development choices with validation data, and estimate final performance with test data. **Learn more:** [Evaluation Basics](./theory/evaluation.html#data-splits).

## U {#u}

### Unsupervised learning {#unsupervised-learning}

Finding structure in data without a provided target label. Clustering is a common example. **Learn more:** [Intro to Machine Learning](./theory/intro_ml.html#unsupervised-learning).
