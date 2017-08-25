
In this module, we learn a bit about deep learning from [Vincent Vanhoucke](http://vincent.vanhoucke.com/), a Principal Scientist 
at Google and Tech Lead in the [Google Brain](https://en.wikipedia.org/wiki/Google_Brain) team.

I won't meticulously be taking notes on the DL stuff here since I've already gone through Vincent's [deep learning course](https://www.udacity.com/course/deep-learning--ud730)
several times in the past year (e.g., by itself, then again when pursuing Udacity's DL nanodegree).

However, for reference's sake, I will definitely include snippets of TensorFlow code.

## Install TensorFlow and say "Hello!"
```
conda create --name=IntroToTensorFlow python=3 anaconda
source activate IntroToTensorFlow
conda install -c conda-forge tensorflow
```
```python
import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```

## Constant Tensors
```python
# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
 # C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
```

## The TF Session
In TF, we first create the neural network graph, then we run it using Session().run().

## The Placeholder
If tf.constant() is a box with a constant in it, and that constant is the same constant for all time,
then you can think of tf.placeholder() as a box that is also used holding constants (e.g., the input/target data), 
but not indefinitely.  It's like public transportation for constant data.

We just need to specify if the placeholder is a plane, train, or automobile -- that is, how big should it be? what
type of data will it hold? should it be able to vary in its carrying capacity?

Only at run time is data put into a placeholder:
```python
x = tf.placeholder(tf.string)
with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```

## Some Operations
```python
x = tf.constant(10,dtype=tf.float32)
y = tf.constant(2,dtype=tf.float32)
z = tf.subtract(tf.divide(x, y), 1)
with tf.Session() as sess:
    print(sess.run(z))
```

### Make it a function w/ placeholders
```python
def tf_op(x, y):
  xx = tf.placeholder(dtype=tf.float32)
  yy = tf.placeholder(dtype=tf.float32)
  z = tf.subtract(tf.divide(x, y), 1)
  with tf.Session() as sess:
    print(sess.run(z, feed_dict={xx: x, yy: y}))
```

## Variables
* So constants are those values that are set in the computation graph before runtime.
    - use case: pi
* Placeholders are like boxes that constants can be put into and taken out of; the box is engineered before runtime,
so that it can deal with shennanigans during runtime.
    - use case:  feature/target data
* A variable is something that is initialized prior to runtime, and subject to change during runtime
    - use case: weights and biases

Since building tensors in TF does not activate the tensor, when one builds a tensor variable, they must explicitly
initialize it be using it in the session:
```python
n_features = 20
n_labels = 2
weight = tf.Variable(tf.truncated_normal((n_features, n_labels)))
bias = tf.Variable(tf.zeros(n_labels))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

# Multi-Digit Classification: 0, 1, 2

First we define some help functions:
```python
# Solution is available in the other "quiz_solution.py" tab
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    # TODO: Return weights
    return tf.Variable(tf.truncated_normal((n_features, n_labels), stddev=0.1))

def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    # TODO: Return biases
    return tf.Variable(tf.zeros(n_labels))


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    # TODO: Linear Function (xW + b)
    return tf.matmul(input,w)+b
    
    
# Here we define a data processing function to extract only a
# limited portion of the MNIST data set s.t. we can experiment quickly.  
def mnist_features_labels(n_labels):
    """
    Gets the first <n> labels from the MNIST dataset
    :param n_labels: Number of labels to use
    :return: Tuple of feature list and label list
    """
    mnist_features = []
    mnist_labels = []

    mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

    # In order to make quizzes run faster, we're only looking at 10000 images
    for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):

        # Add features and labels if it's for the first <n>th labels
        if mnist_label[:n_labels].any():
            mnist_features.append(mnist_feature)
            mnist_labels.append(mnist_label[:n_labels])

    return mnist_features, mnist_labels

```

And, finally, here is where we build the simple neural network:
* we first specify things about the data: the number of features and number of labels
* we use this info to build data tf.placeholder tensors and weight/bias tf.Variable tensors
* we then use the data and weight/bias tensors to construct a linear transformation (the logits)
* at this point, we can choose a learning rate, initialize tf.Variable tensors, and plug the data into a TF session
* within the session, we compute the network's current predictions (softmax of the logits) and the associated loss (using cross entropy)
* to learn from our mistakes, we plug the learning rate and loss into the tf.GradientDescentOptimizer
* Note that we are not really using stochastic gradient descent (SGD) here, as we plug our entire data set in all at once

```python
# Number of features (28*28 image is 784 features)
n_features = 784
# Number of labels
n_labels = 3

# Features and Labels
features = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32)

# Weights and Biases
w = get_weights(n_features, n_labels)
b = get_biases(n_labels)

# Linear Function xW + b
logits = linear(features, w, b)

# Training data
train_features, train_labels = mnist_features_labels(n_labels)

init = tf.global_variables_initializer()
with tf.Session() as session:
    # TODO: Initialize session variables
    session.run(init)
    
    # Softmax
    prediction = tf.nn.softmax(logits)

    # Cross entropy
    # This quantifies how far off the predictions were.
    # You'll learn more about this in future lessons.
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

    # Training loss
    # You'll learn more about this in future lessons.
    loss = tf.reduce_mean(cross_entropy)

    # Rate at which the weights are changed
    # You'll learn more about this in future lessons.
    learning_rate = 0.08

    # Gradient Descent
    # This is the method used to train the model
    # You'll learn more about this in future lessons.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Run optimizer and get loss
    _, l = session.run(
        [optimizer, loss],
        feed_dict={features: train_features, labels: train_labels})

# Print loss
print('Loss: {}'.format(l))

```


## Softmax
s(y[i]) = exp(y[i]) / Sum{j}(exp(y[j]))

Softmax allows us to assign a probability to each label.  If you are wondering why it's called softmax,
then you're in good company.  My speculation is that a "hardmax" would be to assign a "1" to a single label
and a "0" to all other labels (this is oftentimes called "argmax").

```python
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # TODO: Compute and return softmax(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

logits = [3.0, 1.0, 0.2]
print(softmax(logits))
```

Or, in TensorFlow:
```python
with tf.Session() as sess:  print(sess.run(tf.nn.softmax(arr)))
```

## One-Hot Encoding
Example: 
* say you have a categorical target variable, animal, which draws from {cat, dog, fish}
* we want to make this numerical, but not arbitrarily numerical
    - for example, assigning integers s.t. {cat, dog, fish} = {0, 1, 2} is arbitrarily numerical since it shows order where there is none
* one-hot encoding the animals space gives:  {cat, dog, fish} = {[1,0,0], [0,1,0], [0,0,1]}

Pandas, Sklearn, TensorFlow -- they all have ways of one-hot encoding for you, so no worries about re-implementing it 
yourself every time you start a project.  (That said, it's relatively easy to do so!)  

One-hot encoding is the preferred choice when you don't have too many labels.  If you have 1000's of labels, however,
it gets computationally prohibitive and people often turn to embedding vectors instead of one-hot vectors.

You can also picture that 1000's of labels can cause one to suffer from the curse of dimensionality.  Say you have
a categorical feature, e.g., say the animals tensor is an input feature.  As a categorical variable, it is like 
a one parameter feature space... But one-hot encoding it gives back a 3-parameter feature space.  What if you had
10k animal instances to draw from?  One-hot encoding would give back a 10-dimensional feature space.  Such a feature
space could be sparse AF, and almost impossible to learn from...  That's where embedded vectors come in (but we'll
get to 'em later in the course).

For now, we'll stick w/ one-hot encoding.

## Cross Entropy


