---
title: "A friendly introduction to machine learning"
excerpt_separator: "<!--more-->"
categories:
  - Blog
tags:
  - Machine Learning
  - AI
  - Beginners
---

Ever wondered how your phone can recognize your face or how Google sorts spam emails? Behind these marvels is a field called **machine learning**, where algorithms learn patterns from data to make predictions or decisions.

In this post, I’ll explore the three main types of machine learning, the importance of preprocessing, and how simple models like the **perceptron** and **logistic regression** work under the hood.

## Types of machine learning

### Supervised Learning

You have labelled data and want to predict the label of future data (Direct feedback with error function) E.g. Image and speech recognition, recommendation systems, fraud detection…

Consider email spam filtering, we can train a model on a corpus of emails correctly labelled as spam and not spam.

A supervised learning problem with discrete labels (e.g. spam or no spam) is called a **classification task.** You often have multi-class classification such as digit recognition (0,1,2…) or letter recognition (A,B,C…).

Another subcategory of supervised learning is **regression**. In regression you want to build a model (e.g. linear) for some data and use it to predict the output of an input. The output labels are continuous.

NOTE: In machine learning the input variables are called features and the output variables are called targets.

### Unsupervised Learning

Data with no labels and want to find structure in the data

E.g. Given a set of news articles found on the web, group them into a set of articles about the same topic; like weather, crime, social media…

In Unsupervised Learning (UL) we are dealing with unlabelled data and wish to find structure in it; without the guidance of any reward function.

### Reinforcement Learning

Make decisions to maximise reward (can be delayed reward)

E.g. Learn to play a game against an opponent

In reinforcement learning (RL) an agent is that improves its performance on some tasks based on interactions with the environment. This learning is led by some reward function that guides the agent towards better performance by maximising the reward. 

The main issue in RL is the reward may be delayed, so the agent has no idea what moves it did lead to success/failure (Delayed reward problem).

![image.png](/assets/2025-06-26-Intro-to-ML/image.png)

There are many subtypes of reinforcement learning.

---

**NOTE:** Loss function, Cost function and Error function all refer to the same thing. In some literature the term loss is reserved for the loss of a single datum and the cost is reserved for the loss of the entire dataset.

---

## Preprocessing

Raw data rarely comes in the shape/form adequate for training a model. So preprocessing is simply a term for any alteration to raw data for making it more suitable as a training dataset.

Many machine learning algorithms require the data to be on the same scale, often achieved by transforming the data into the range $[0, 1]$ or a standard normal distribution ($\mu = 0, \space \sigma=1$)

Some of the data may be highly correlated so dimensionality reduction may be applied to project the data onto a subspace, which reduces the storage required and makes the learning algorithm can run much faster.

This can also improve predictive performance as some of the “irrelevant” features or noise is removed.

This is in the hopes of improving the training performance and reducing the number of parameters in the model. This helps reduce overfitting of the model on the training data.

We also hold onto some of the dataset for testing the model on data that’s not been used to train it.

## Selecting and training a predictive model

There are many machine learning algorithms. It’s you job to select the appropriate algorithm for your purposes and not just stick to one for all your purposes. This quote sums it up well:

> *I suppose it is tempting, if the only tool you have is a hammer, to treat everything as if it were a nail* (Abraham Maslow, 1966)
> 

For example, each classification algorithm has its advantages and disadvantages, there’s no **one** algorithm that enjoys superiority if we don’t make assumptions about the data.

It’s essential to compare at least a handful of training models to select and train the best performing model.

But before training a model we need to decide upon a **metric** to measure its performance. 

Also **DO NOT** assume the default parameters of a ML algorithm are optimal for our specific task. We will use **hyper-parameter optimisation** to fine tune the performance of our model.

---

# Simple ML algorithms for classification

We will start with the earliest ML algorithms for classification: the perceptron.

## Artificial neurons

The basic idea of an artificial neuron is simply a function that gives an output if the neuron is “activated” and none otherwise. in practice this is done with an activation function $\sigma(z)$, the earliest $\sigma$ used is a simple step function with bias $\theta$:

$$
\sigma(z)= \begin{cases}1  &\text{if} \space{}z\ge\theta\\ 0 &\text{otherwise} \end{cases}
$$

where $z$ is a function of the input variables $\underline{x}$  ($z=\underline{w}\cdot\underline{x}$).

In practice we need to learn both $w$ (the weight vector) and $\theta$, so it’s useful to redefine both $\sigma$ and $z$:

$$
\sigma(z)= \begin{cases}1  &\text{if} \space{}z\ge0\\ 0 &\text{otherwise} \end{cases}\\ z=\underline{w}\cdot\underline{x}+b
$$

Where $b=-\theta$ is the bias.

## Perceptron

The basic idea of the perceptron is to define a loss function L; that quantifies the difference between the model’s outputs and the targets. We then minimise L using gradient descent.

For a very simple single neuron model $y_{i}=\sigma(z)$ we could define L as:

$$
L =\frac{1}{n}\sum_{i=1}^n\frac{1}{2}(y^i-y^i)^2
$$

Where $y_{i}$ is the target and $y_{i}$ is the output of our model. The sum is over the entire dataset.

This loss function is called the **Mean-Squared-Error.**

To apply gradient descent we need to find $\frac{\partial L}{\partial \underline{w}}, \frac{\partial L}{\partial b}$:

$$
\frac{\partial L}{\partial \underline{w}} = \frac{1}{n}\sum_{i=1}^n(y^i-y^i)\frac{\partial y^i}{\partial\underline{w}} = \frac{1}{n}\sum_{i=1}^n(y^i-y^i)\frac{\partial \sigma(z^i)}{\partial z^i}\frac{\partial z^i}{\partial \underline{w}} =\frac{1}{n}\sum_{i=1}^n(y^i-y^i)\sigma'(z^i)\underline{x^i}
$$

Similarly:

$$
\frac{\partial L}{\partial b}= \frac{1}{n}\sum_{i=1}^n(y^i-y^i)\sigma'(z^i)
$$

We then update $\underline{w}$ and $b$, with learning rate $\eta$:

$$
\Delta \underline{w} = -\eta\frac{\partial L}{\partial \underline{w}} = -\frac{\eta}{n}\sum_{i=1}^n(y^i-y^i)\sigma'(z^i)\underline{x^i}
$$

$$
\Delta b = -\eta\frac{\partial L}{\partial b} = -\frac{\eta}{n}\sum_{i=1}^n(y^i-y^i)\sigma'(z^i)
$$

We now run into a major issue $\sigma'(z^i)$ is zero for all $z^i$, so that $\Delta \underline{w},\Delta b=0$.

Our learning has been stalled!

> Actually  $\sigma'(z^i)=\delta(z^i)$ where $\delta$ is the Dirac delta function, but $z^i=0$ is an unimportant edge case
> 

This can be fixed by simply removing $\sigma'(z^i)$:

$$
\Delta \underline{w} = -\frac{\eta}{n}\sum_{i=1}^n(y^i-y^i)\underline{x^i}
$$

$$
\Delta b =  -\frac{\eta}{n}\sum_{i=1}^n(y^i-y^i)
$$

The convergence of the perceptron is only guaranteed if the data is linearly separable:

![image.png](/assets/2025-06-26-Intro-to-ML/image1.png)

If two classes can’t be linearly separated we can set a maximum number of passes over the training dataset (epochs), otherwise the model would keep training indefinitely. We could also set a maximum number of misclassified data points.

There are also algorithms that allows for some leeway in the decision boundary (e.g. the Adaline algorithm), and converges even if the two classes aren’t linearly separable.

## **Improving gradient descent through feature scaling**

Many machine learning algorithms require some sort of feature scaling; gradient descent is such an algorithm.

To do so we **standardise** the dataset, so that the $j_{th}$ **feature** is replaced with:

$$
x_j'=\frac{x_j-\mu_j}{\sigma_j}
$$

The new data points $x_j'$ have mean of 0 and a standard deviation of 1.

A reason standardisation helps gradient descent is that it makes it easier to find a learning rate $\eta$ that works all features. If the features are vastly different sizes it’s difficult to find a learning rate that works for all features simultaneously.

## **Stochastic gradient descent**

In the perceptron algorithm we used [updates](https://www.notion.so/Machine-Learning-ff5c393511054d4eb253d9b52ae9f6df?pvs=21) that required us to iterate over the entire training set. This can be very computationally expensive if our training set consists of a very large number of datapoints. Stochastic gradient descent (SGD) trains the model using single data points at a time.

This boils down to replacing the updates to:

$$
\Delta \underline{w} = -\eta\frac{\partial L}{\partial \underline{w}} = -\eta(y^i-y^i)\sigma'(z^i)\underline{x^i}
$$

$$
\Delta b = -\eta\frac{\partial L}{\partial b} = -\eta(y^i-y^i)\sigma'(z^i)
$$

Where $i$ is the index of a randomly chosen datapoint.

Although this gives less reliable updates in practice the model will train faster due to the larger number of updates possible.

### Minibatch gradient descent

A compromise between full batch gradient descent and SGD is minibatch gradient descent. Minibatch gradient descent **randomly** samples a few points in the training set (called a **minibatch**) to estimate the updates to the model’s parameters. 

This boils down to replacing the updates to:

$$
\Delta \underline{w} = -\eta\frac{\partial L}{\partial \underline{w}} = -\frac{\eta}{n_{\text{batch}}}\sum_{\text{batch}}(y^i-y^i)\sigma'(z^i)\underline{x^i}
$$

$$
\Delta b = -\eta\frac{\partial L}{\partial b} = -\frac{\eta}{n_\text{batch}}\sum_{\text{batch}}(y^i-y^i)\sigma'(z^i)
$$

Where $n_\text{batch}$ is the number of datapoints in our minibatch. 

To prevent using the same batch we shuffle the training set at the start of each epoch.

# Training a supervised ML algorithm

The five main steps when when training a supervised ML algorithm are:

1. Select features (e.g. height, weight, size etc…) and collecting labelled training examples
2. Choose a performance metric (e.g. SME)
3. Choose a learning algorithm (e.g. minibatch gradient descent) and train a model
4. Evaluate the performance of the model
5. Change the settings of the algorithm(e.g. learning parameter) and tune the model (e.g number of neurons)

## Training a perceptron with scikit-learn

We will use the iris dataset, which gives properties of some flowers (the features) and their names (the targets). 

There are 3 flower types: Setosa, Versicolour and Virginica. 

Each denoted with the numbers: 1, 2, 3 respectively.

---

We will first import the dataset using sklearn:

```python
from sklearn import datasets

iris = datasets.load_iris()
features = iris.data[:, [2, 3]]
targets = iris.target
```

Where we restricted the feature set to just the 2nd and 3rd feature.

---

We now will split the dataset into training and testing:

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, targets, 
test_size=0.3, random_state=1, stratify=targets)
```

For reproducibility of the code we’ve used random_state = 1 which ensures the data is split the same way every time.

To make sure that each flower type is represented equally we stratify the data so that we an equal number of each flower type in the training and testing sets.

---

We now [scale the features](https://www.notion.so/Machine-Learning-ff5c393511054d4eb253d9b52ae9f6df?pvs=21) so that they have $\mu=0,\space{} \sigma=1$:

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

We first create an instance of StandardScaler sc. 

By calling the “fit” method we can store the mean and standard deviation of the data into sc and then transform the features into the desired form.

---

We now create our model (perceptron) and train it:

```python
from sklearn.linear_model import Perceptron

model = Perceptron(eta0=0.1, random_state=1)
model.fit(X_train_std, y_train)
```

The learning rate eta0 has been chosen so that the model will train correctly.

random_state = 1 ensures the same shuffling of the dataset at each epoch.

---

We now test how well the model generalises to the test data (generalisation accuracy):

```python
y_test_pred = model.predict(X_test_std) != y_test
print(f"Misclassified = {y_test_pred.sum()}/{len(y_test_pred)}")
```

y_test_pred stores whether each example’s been correctly classified in a boolean (using the predict method).

Executing the code shows 1/45 flowers are misclassified.

It is standard to quote the accuracy of the model as a percentage of correctly classified examples: 97.8%.

---

Unfortunately, by plotting the data you can see that the flower types aren’t linearly separable:

![image.png](/assets/2025-06-26-Intro-to-ML/image2.png)

Thus, the perceptron algorithm won’t converge in this case, more powerful linear classifiers are needed, that can converge to loss minimums. 

## Logistic regression

NOTE: Logistic regression is a model for classification **NOT** regression.

Logistic regression is one of the most widely used algorithms for classification in industry.

Although you can use logistic regression for multiple classes (multinomial logistic regression) we will only cover binary classification.

To discuss logistic regression we first introduce the notion of **odds.** Given a probability p for an event, the odds of that event is $\frac{p}{1-p}$. We can further define the logit function as:

$\text{logit}(p)=\text{log}(\frac{p}{1-p})$ also called the log-odds. 

Under the logistic model we assume there is a linear relationship between z and the log-odds:

$$
\text{logit}(p)=z=\underline{w}\cdot\underline{x}+b \text{ }
$$

Which can be inverted to give:

$$
p=\frac{1}{1+e^{-z}}
$$

Known as the sigmoid function, it will be used as the activation function of our neurons.

The output of out neuron will be $\sigma(z)=1/(1+e^{-z})$ and it represents the probability of being in a particular class in our binary classification problem. 

We define the output $y^i$ as:

$$
y=\begin{cases}1&if \text{ }\sigma(z)\ge0.5 \\ 0 & \text{otherwise}\end{cases}
$$

Which is equivalent to:

$$
y=\begin{cases}1&if \text{ }z\ge0 \\ 0 & \text{otherwise}\end{cases}
$$

We want to maximise the **likelihood $\mathcal{L}$** that our classification matches the **true** classification:

$$
\mathcal{L}=\prod_{i=1}^n p(y^i|\underline{x}^i;\underline{w},b)
$$

Now the probability $p(y^i|\underline{x}^i;\underline{w},b)$ is given by $\sigma(z)$ if $y^i=0$ and  $1-\sigma(z)$ if $y^i=1$.

This can be written as $\sigma(z^i)^{y^i}(1-\sigma(z^i))^{1-y^i}$.

[Convince yourself this is true, by checking the $y^i=0,1$ cases].

$$
\mathcal{L}=\prod_{i=1}^n \sigma(z^i)^{y^i}(1-\sigma(z^i))^{1-y^i}
$$

It is now useful to introduce a logarithm and reduce the **log-likelihood**:

$$
\log\mathcal{L}=\sum_{i=1}^n y^i\log\sigma(z^i)+(1-y^i)\log(1-\sigma(z^i))
$$

We now notice the problem of maximising log-likelihood can be remapped to minimising a loss function L:

$$
L=-\frac{1}{n}\sum_{i=1}^n y^i\log\sigma(z^i)+(1-y^i)\log(1-\sigma(z^i))
$$

By using $\sigma'(z)=\sigma(z)(1-\sigma(z))$ we can show that the updates to $\underline{w}, b$ are given by:

$$
\Delta \underline{w} = -\frac{\eta}{n}\sum_{i=1}^n(y^i-y^i)\underline{x^i}
$$

$$
\Delta b =  -\frac{\eta}{n}\sum_{i=1}^n(y^i-y^i)
$$

Which is exactly what we had for our perceptron algorithm! Except now $y^i$ is now worked out using a sigmoid function rather than a step function.

### Training a logistic regression model with scikit-learn

The data will be preprocessed exactly as it was in the [perceptron](https://www.notion.so/Machine-Learning-ff5c393511054d4eb253d9b52ae9f6df?pvs=21) section.

```python
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
model_lr.fit(X_train_std, y_train)
```

Ignore C=100.0, this is a separate technique to reduce overfitting, will be discussed in the [regularisation section](https://www.notion.so/Machine-Learning-ff5c393511054d4eb253d9b52ae9f6df?pvs=21). 

NOTE: For minimising convex loss functions, such as the logistic regression loss, it is recommended to use more advanced approaches than SGD. Sklearn implements a range of optimisation algorithms, which can be selected via the solver parameter, namely: 'newton-cg', 'lbfgs', 'liblinear', 'sag', and 'saga'.

When you check the classification accuracy you see nothing has changed it has remained the same: 97.8%. So on the surface we’ve had no change, however if you look at the decision boundaries:

![image.png](/assets/2025-06-26-Intro-to-ML/image3.png)

They are completely different! Arguably better?

## Regularisation

Overfitting is a big problem in ML, where a model won’t generalise to the testing data and simply learns the nuances of the training data, to get a lower loss on it.

Regularisation is a class of techniques that tackle this issue.

The most common form of regularisation is **L2 regularisation;** which involves adding an additional term to the loss function:

$$
L=-\frac{1}{n}\sum_{i=1}^n y^i\log\sigma(z^i)+(1-y^i)\log(1-\sigma(z^i))+\frac{\lambda}{2n}|\underline{w}|^2
$$

Here $\lambda$ is the **regularisation parameter** and the n added to scale the regularisation term similar to the loss term.

This changes the updates to:

$$
\Delta \underline{w} = -\frac{\eta}{n}\sum_{i=1}^n(y^i-y^i)\underline{x^i}-\frac{\lambda}{n}\underline{w}
$$

$$
\Delta b =  -\frac{\eta}{n}\sum_{i=1}^n(y^i-y^i)
$$

Clearly L2 regularisation acts to reduce the parameters in $\underline{w}$.

The parameter C we previously saw comes from a convention in support vector machines, and it’s inversely proportional to $\lambda$.

So if L2 regularisation is so good why not always use it? 

We’ve got to be careful, if $\lambda$ is too small we won’t see much effect from the regularisation term, however is $\lambda$ is too large we will be driving our weight vector $\underline{w}$ towards $\underline{0}$, hampering our classification accuracy and underfitting our data.