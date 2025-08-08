---
layout: post
title:  Artificial Neural Network
parent: Minis
nav_order: 3
---

## Artificial Neural Network

(based on [this textbook](https://www.google.com/books/edition/Grokking_Machine_Learning/fNhOEAAAQBAJ?hl=en&gbpv=0))

If you haven't learned how ANNs work, check out [this](https://astarryknight.github.io/ai-ml/src/theory/ann.html) lesson!

---

## A Very Simple Neural Network
The beginning of the program just defines libraries and the values of the parameters, and creates a list which contains the values of the weights that will be modified (those are generated randomly).

```python
import numpy, random, os
lr = 1 #learning rate
bias = 1 #value of bias
weights = [random.random(),random.random(),random.random()] #weights generated in a list (3 weights in total for 2 neurons and the bias)
```

Here we create a function which defines the work of the output neuron. It takes 3 parameters (the 2 values of the neurons and the expected output). “outputP” is the variable corresponding to the output given by the Perceptron. Then we calculate the error, used to modify the weights of every connections to the output neuron right after.

```python
def Perceptron(input1, input2, output) :
   outputP = input1*weights[0]+input2*weights[1]+bias*weights[2]
   if outputP > 0 : #activation function (here Heaviside)
      outputP = 1
   else :
      outputP = 0
   error = output - outputP
   weights[0] += error * input1 * lr
   weights[1] += error * input2 * lr
   weights[2] += error * bias * lr
```

We create a loop that makes the neural network repeat every situation several times. This part is the learning phase. The number of iteration is chosen according to the precision we want. However, be aware that too much iterations could lead the network to over-fitting, which causes it to focus too much on the treated examples, so it couldn’t get a right output on case it didn’t see during its learning phase.

However, our case here is a bit special, since there are only 4 possibilities, and we give the neural network all of them during its learning phase. A Perceptron is supposed to give a correct output without having ever seen the case it is treating.

```python
for i in range(50) :
   Perceptron(1,1,1) #True or true
   Perceptron(1,0,1) #True or false
   Perceptron(0,1,1) #False or true
   Perceptron(0,0,0) #False or false
```

Finally, we can ask the user to enter the values to check if the Perceptron is working. This is the testing phase.

The activation function Heaviside is interesting to use in this case, since it takes back all values to exactly 0 or 1, since we are looking for a false or true result. We could try with a sigmoid function and obtain a decimal number between 0 and 1, normally very close to one of those limits.

```python
x = int(input())
y = int(input())
outputP = x*weights[0] + y*weights[1] + bias*weights[2]
if outputP > 0 : #activation function
   outputP = 1
else :
   outputP = 0
print(x, "or", y, "is : ", outputP)
```

We could also save the weights that the neural network just calculated in a file, to use it later without making another learning phase. It is done for way bigger project, in which that phase can last days or weeks.

```python
outputP = 1/(1+numpy.exp(-outputP)) #sigmoid function
```

Here's some more fun with neural networks on [Tensorflow Playground](https://playground.tensorflow.org/)

---

## ANN for Classification

### Processing the Data
```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

Download the dataset [here](../datasets/Churn_Modelling.csv)

```python
path = '../datasets/spotify-2023.csv' #replace this with your path/to/dataset
df = pd.read_csv(path)
```

```python
dataset.head(25)
```

| RowNumber | CustomerId | Surname | CreditScore | Geography | Gender | Age | Tenure    | Balance | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited | 
| -- | --------- | ---------- | ------- | ----------- | --------- | ------ | --- | --------- | ------- | ------------- | --------- | -------------- | --------------- | ------ | - | - | - | - | - | - | - | - | - | - | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 1  | 15634602  | Hargrave   | 619     | France      | Female    | 42     | 2   | 0.00      | 1       | 1             | 1         | 101348.88      | 1               |
| 2  | 15647311  | Hill       | 608     | Spain       | Female    | 41     | 1   | 83807.86  | 1       | 0             | 1         | 112542.58      | 0               |
| 3  | 15619304  | Onio       | 502     | France      | Female    | 42     | 8   | 159660.80 | 3       | 1             | 0         | 113931.57      | 1               |
| 4  | 15701354  | Boni       | 699     | France      | Female    | 39     | 1   | 0.00      | 2       | 0             | 0         | 93826.63       | 0               |
| 5  | 15737888  | Mitchell   | 850     | Spain       | Female    | 43     | 2   | 125510.82 | 1       | 1             | 1         | 79084.10       | 0               |
| 6  | 15574012  | Chu        | 645     | Spain       | Male      | 44     | 8   | 113755.78 | 2       | 1             | 0         | 149756.71      | 1               |
| 7  | 15592531  | Bartlett   | 822     | France      | Male      | 50     | 7   | 0.00      | 2       | 1             | 1         | 10062.80       | 0               |
| 8  | 15656148  | Obinna     | 376     | Germany     | Female    | 29     | 4   | 115046.74 | 4       | 1             | 0         | 119346.88      | 1               |
| 9  | 15792365  | He         | 501     | France      | Male      | 44     | 4   | 142051.07 | 2       | 0             | 1         | 74940.50       | 0               |
| 10 | 15592389  | H?         | 684     | France      | Male      | 27     | 2   | 134603.88 | 1       | 1             | 1         | 71725.73       | 0               |
| 11 | 15767821  | Bearce     | 528     | France      | Male      | 31     | 6   | 102016.72 | 2       | 0             | 0         | 80181.12       | 0               |
| 12 | 15737173  | Andrews    | 497     | Spain       | Male      | 24     | 3   | 0.00      | 2       | 1             | 0         | 76390.01       | 0               |
| 13 | 15632264  | Kay        | 476     | France      | Female    | 34     | 10  | 0.00      | 2       | 1             | 0         | 26260.98       | 0               |
| 14 | 15691483  | Chin       | 549     | France      | Female    | 25     | 5   | 0.00      | 2       | 0             | 0         | 190857.79      | 0               |
| 15 | 15600882  | Scott      | 635     | Spain       | Female    | 35     | 7   | 0.00      | 2       | 1             | 1         | 65951.65       | 0               |
| 16 | 15643966  | Goforth    | 616     | Germany     | Male      | 45     | 3   | 143129.41 | 2       | 0             | 1         | 64327.26       | 0               |
| 17 | 15737452  | Romeo      | 653     | Germany     | Male      | 58     | 1   | 132602.88 | 1       | 1             | 0         | 5097.67        | 1               |
| 18 | 15788218  | Henderson  | 549     | Spain       | Female    | 24     | 9   | 0.00      | 2       | 1             | 1         | 14406.41       | 0               |
| 19 | 15661507  | Muldrow    | 587     | Spain       | Male      | 45     | 6   | 0.00      | 1       | 0             | 0         | 158684.81      | 0               |
| 20 | 15568982  | Hao        | 726     | France      | Female    | 24     | 6   | 0.00      | 2       | 1             | 1         | 54724.03       | 0               |
| 21 | 15577657  | McDonald   | 732     | France      | Male      | 41     | 8   | 0.00      | 2       | 1             | 1         | 170886.17      | 0               |
| 22 | 15597945  | Dellucci   | 636     | Spain       | Female    | 32     | 8   | 0.00      | 2       | 1             | 0         | 138555.46      | 0               |
| 23 | 15699309  | Gerasimov  | 510     | Spain       | Female    | 38     | 4   | 0.00      | 1       | 1             | 0         | 118913.53      | 1               |
| 24 | 15725737  | Mosman     | 669     | France      | Male      | 46     | 3   | 0.00      | 2       | 0             | 1         | 8487.75        | 0               |
| 25 | 15625047  | Yen        | 846     | France      | Female    | 38     | 5   | 0.00      | 1       | 1             | 1         | 187616.16      | 0               |



**Creating feature and target vectors**

Looking at the features we can see that RowNumber, CustomerId, and Surname will have no relation with a customer leaving the bank. We drop them from X, which now contains the features indices from 3 to 12.

```python
X = dataset.iloc[:, 3:13].values
```

```python
y = dataset.iloc[:, 13].values

#Printing out the values of X --> Which contains the features
#                           y --> Which contains the target variable
print(pd.DataFrame(X[:10]))
print()
print(pd.DataFrame(y[:10]))
```

```markdown
    0        1       2   3  4          5  6  7  8          9
0  619   France  Female  42  2        0.0  1  1  1  101348.88
1  608    Spain  Female  41  1   83807.86  1  0  1  112542.58
2  502   France  Female  42  8   159660.8  3  1  0  113931.57
3  699   France  Female  39  1        0.0  2  0  0   93826.63
4  850    Spain  Female  43  2  125510.82  1  1  1    79084.1
5  645    Spain    Male  44  8  113755.78  2  1  0  149756.71
6  822   France    Male  50  7        0.0  2  1  1    10062.8
7  376  Germany  Female  29  4  115046.74  4  1  0  119346.88
8  501   France    Male  44  4  142051.07  2  0  1    74940.5
9  684   France    Male  27  2  134603.88  1  1  1   71725.73

   0
0  1
1  0
2  1
3  0
4  0
5  1
6  0
7  1
8  0
9  0
```

**Encoding categorical data**

Neural networks can only handle numerical data. The categorical data in Geography and Gender won't work. You might recall there was a concept called get_dummies in pandas that would convert categorical variables to several binary columns instead. That's what we'll do here, but with something called LabelEncoder.

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
```

```python
dataset['Geography'].unique()
```

```python
print(pd.DataFrame(X[:10]))
```

```markdown
     0        1       2   3  4          5  6  7  8          9
0  619   France  Female  42  2        0.0  1  1  1  101348.88
1  608    Spain  Female  41  1   83807.86  1  0  1  112542.58
2  502   France  Female  42  8   159660.8  3  1  0  113931.57
3  699   France  Female  39  1        0.0  2  0  0   93826.63
4  850    Spain  Female  43  2  125510.82  1  1  1    79084.1
5  645    Spain    Male  44  8  113755.78  2  1  0  149756.71
6  822   France    Male  50  7        0.0  2  1  1    10062.8
7  376  Germany  Female  29  4  115046.74  4  1  0  119346.88
8  501   France    Male  44  4  142051.07  2  0  1    74940.5
9  684   France    Male  27  2  134603.88  1  1  1   71725.73
```

Creating label encoder object no. 1 to encode Geography name (index 1 in features) for France, Spain, Germany.

Next, encoding Geography from string to just 3 numbers (0 = France, 1 = Spain, 2 = Germany).

```python
ct = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
print(pd.DataFrame(X[:10]))
```

```markdown
    0    1    2    3       4   5  6          7  8  9  10         11
0  1.0  0.0  0.0  619  Female  42  2        0.0  1  1  1  101348.88
1  0.0  0.0  1.0  608  Female  41  1   83807.86  1  0  1  112542.58
2  1.0  0.0  0.0  502  Female  42  8   159660.8  3  1  0  113931.57
3  1.0  0.0  0.0  699  Female  39  1        0.0  2  0  0   93826.63
4  0.0  0.0  1.0  850  Female  43  2  125510.82  1  1  1    79084.1
5  0.0  0.0  1.0  645    Male  44  8  113755.78  2  1  0  149756.71
6  1.0  0.0  0.0  822    Male  50  7        0.0  2  1  1    10062.8
7  0.0  1.0  0.0  376  Female  29  4  115046.74  4  1  0  119346.88
8  1.0  0.0  0.0  501    Male  44  4  142051.07  2  0  1    74940.5
9  1.0  0.0  0.0  684    Male  27  2  134603.88  1  1  1   71725.73
```

Creating label encoder object no. 1 to encode Gender name (now index 4 in features) for Male, Female.

Next, encoding Gender from string to 2 numbers (0 for Male and 1 for Female)

```python
ct = ColumnTransformer([("Gender", OneHotEncoder(), [4])], remainder = 'passthrough')
X = ct.fit_transform(X)
print(pd.DataFrame(X[:10]))
```

```markdown
    0    1    2    3    4    5   6  7          8  9  10 11         12
0  1.0  0.0  1.0  0.0  0.0  619  42  2        0.0  1  1  1  101348.88
1  1.0  0.0  0.0  0.0  1.0  608  41  1   83807.86  1  0  1  112542.58
2  1.0  0.0  1.0  0.0  0.0  502  42  8   159660.8  3  1  0  113931.57
3  1.0  0.0  1.0  0.0  0.0  699  39  1        0.0  2  0  0   93826.63
4  1.0  0.0  0.0  0.0  1.0  850  43  2  125510.82  1  1  1    79084.1
5  0.0  1.0  0.0  0.0  1.0  645  44  8  113755.78  2  1  0  149756.71
6  0.0  1.0  1.0  0.0  0.0  822  50  7        0.0  2  1  1    10062.8
7  1.0  0.0  0.0  1.0  0.0  376  29  4  115046.74  4  1  0  119346.88
8  0.0  1.0  1.0  0.0  0.0  501  44  4  142051.07  2  0  1    74940.5
9  0.0  1.0  1.0  0.0  0.0  684  27  2  134603.88  1  1  1   71725.73
```

We remove the first column because two columns is enough to encode three countries. In other words, if those two columns are both 0, it must be the third country.

```python
X = X[:,1:]
print(pd.DataFrame(X[:10]))
```

```markdown
    0    1    2    3    4   5  6          7  8  9  10         11
0  0.0  1.0  0.0  0.0  619  42  2        0.0  1  1  1  101348.88
1  0.0  0.0  0.0  1.0  608  41  1   83807.86  1  0  1  112542.58
2  0.0  1.0  0.0  0.0  502  42  8   159660.8  3  1  0  113931.57
3  0.0  1.0  0.0  0.0  699  39  1        0.0  2  0  0   93826.63
4  0.0  0.0  0.0  1.0  850  43  2  125510.82  1  1  1    79084.1
5  1.0  0.0  0.0  1.0  645  44  8  113755.78  2  1  0  149756.71
6  1.0  1.0  0.0  0.0  822  50  7        0.0  2  1  1    10062.8
7  0.0  0.0  1.0  0.0  376  29  4  115046.74  4  1  0  119346.88
8  1.0  1.0  0.0  0.0  501  44  4  142051.07  2  0  1    74940.5
9  1.0  1.0  0.0  0.0  684  27  2  134603.88  1  1  1   71725.73
```

Splitting the dataset into the Training set and Test set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

---
### Let's Make an ANN!

Let's list out the steps involved in training the ANN with Stochastic Gradient Descent.

1) Randomly initialize the weights to small numbers close to but not 0.

2) Input the 1st observation of your dataset in the input layer, with each feature in one input node.

3) Forward-Propagation from left to right. The neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until getting the predicted result y.

4) Compare the predicted result with the actual result. Measure the generated error.

5) Back-Propagation: From right to left, error is back propagated. Update the weights according to how much they are responsible for the error. The learning rate tells us by how much we should update the weights.

6) Repeat steps 1 to 5 and update the weights after each observation (reinforcement learning). Or: Repeat Steps 1 to 5, but update the weights only after a batch of observations (batch learning)

7) When the whole training set is passed through the ANN, that completes an epoch. Redo more epochs.

```python
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # For building the Neural Network layer by layer
from keras.layers import Dense # To randomly initialize the weights to small numbers close to 0(But not 0)
```

**Initializing the ANN**

We will not put any parameter in the sequential object since we will be defining the layers manually.

```python
classifier = Sequential()
```

**Adding the input layer and the first hidden layer**

How many nodes of the hidden layer do we actually need? There is no rule of thumb, but you can set the number of nodes in hidden layers as an average of the number of nodes in input and output layers, respectively. Here avg= (11+1)/2==>6 So set output dim=6

The activation Function is Rectifier Activation Function.

The kernel initializer will initialize the hidden layer weights uniformly.

Input dim tells us the number of nodes in the input layer. This is done only once and won't be specified in further layers.

```python
classifier.add(Dense(activation="relu", input_dim=12, units=6, kernel_initializer="uniform"))
```

Adding the second hidden layer
```python
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
```

Adding the output layer

The sigmoid activation function is used whenever we need the probabilities of 2 categories (similar to logistic regression). We switch to Softmax activation functions when the dependent variable has more than 2 categories.

```python
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
```

Compiling the ANN

The Adam optimizer is a form of stochastic gradient descent. Luckily for you, this does all of the math behind the scenes.

```python
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

**Fitting the ANN to the Training set**

This step will take some time for large epoch values. A batch size of 10 means that the weights will update after every 10 observations. Epoch is a round of whole data flow through the network.

```python
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
```

---

### Making predictions and evaluating the model

```python
y_pred = classifier.predict(X_test)
```

If y_pred is larger than 0.5, it returns true (1). Otherwise false (2). This determines what the neural network "voted" regarding this customer.

```python
y_pred = (y_pred > 0.5)
```

Making the Confusion Matrix.
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
tn, fp, fn, tp = cm.ravel()
accuracy = (tn + tp) / (tn + tp + fn + fp)
print(accuracy)
```

```markdown
[[1538   57]
 [ 258  147]]
0.8425
```

**Task: Play around with the batch size and epoch hyperparameters. Compare the accuracies**


---

### Predicting whether a new customer will stay at the bank

Input: Two binary columns for geography, two binary columns for gender, credit score, age, tenure, balance, num products, has credit card, is active member, estimated salary

```python
#0 0 1 0 619 42 2 0 1 1 1 56700
new_customer = [[0, 0, 1, 0, 619, 42, 2, 5000, 1, 1, 0, 50700]]
new_customer = sc.transform(new_customer)
new_prediction = classifier.predict(new_customer)
print(new_prediction)
```

```markdown
[[0.5705499]]
```