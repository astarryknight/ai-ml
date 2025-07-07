---
layout: post
title:  2. Optimization & Gradient Descent
parent: Theory
nav_order: 3
---

## Linear Regression
The most basic machine learning model is linear regression. This, like any other regrssion model, predicts for a continuous value. Since this is the most simple learning model, let's try to build it!

But first, what is linear regression?

"Linear regression is a statistical model which estimates the **linear relationship** between a **scalar response** and **one or more explanatory variables**"

Basically, linear regression is a way to predict a single target variable by using the linear relationship between that target variable and the rest of the features in the dataset. The goal of linear regression is to draw a line closest to all data points.

There is actually a really easy way to calculate the regression line, or line of best-fit, numerically, but let's try to use more of an machine-learning approach to linear regression. Let's look back at our 3-Step Framework. One thing that you may have noticed is that we did not account for some way to change our model in response to errors in our predictions. We make changes to our model to reduce error in a process called **optimization**. One way we can optimize our models is through *gradient descent*.

## Optimization: Gradient Descent
![Gradient Descent Figure](../res/grad_desc_vis.png)

**Gradient** - a vector that points in the direction of the steepest increase of a function at a specific point

For a function of multiple variables, such as $f(x, y, z)$, the gradient is represented as the vector $\nabla f$, and its components are:

$
\nabla f = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right]$

$x$