
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
I think we already talked about this um-teen times in these notes... Or maybe those were notes for my other courses.
* cross entropy is not symmetric in its arguments
* however, cross entropy turns logistic regression into a convex optimization problem
* cross entropy is the standard when generalizing simple logistic regression to multinomial logistic classification 
* in the multinomial setting, XE is summed over all possibilities for a single data point
    - if computing the loss over a mini-batch, one then sums this sum over all data points in the batch

<img src=./images/xe.png width=400>
<img src=./images/xe-loss.png width=400>


### XE in TF
For a single data point with several possible classifications:
```python
import tensorflow as tf

# Pretend we've already done the forward pass on a single data point to 
#     get softmax_data and are now comparing it to the one-hot-encoded target 
#     for this data point, one_hot_data
softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

# Create TF Tensors for the data
#   -- this seems redundant when computing for just a single data point,
#      but the importance of the "bucket nature" of placeholders becomes more 
#      obvious if/when we compute over many data points that need to pass through
#      the same graph
softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# Create the XE Tensor in the graph
xe = -tf.reduce_sum(one_hot * tf.log(softmax))

# Run time: Compute the XE 
with tf.Session() as sess:
    output = sess.run(xe, feed_dict={softmax: softmax_data, one_hot: one_hot_data})
    print(output)
```

## Feature Normalization & Weight Initialization
For numerical stability, it is best to work with numbers that are all in the same ballpark, e.g.,
having features with a scalelength of order 1e-6 mixed with those of order 1e6 is not a great idea.
To be well-conditioned, the numbers should also cluster around zero (if this is not the case, the problem
is poorly conditioned).

One common way to try to achieve this is to use features with zero mean and unit variance:
```
f_new = (f_old - mean(f_old)) / stdev(f_old)
```

For RGB images where pixel values range between 0-255:
```
pix_new = (pix_old - 128) / 128
```

Similarly, we want our weights to be initialized in the same [-1,1] ballpark.  One way to do this
is to draw weights from a normal distribution.  A slightly better way is to draw from a truncated
normal distribution.  This is because large weight values generally mean that the network has 
confidence in the associated features, so we do not want randomly high numbers from the normal distro
to be falsely interpreted by the network.  In the same way, we do not want to use a normal distirubtion
that has a large deviation --- even unit variance is often too big.  

If you are only working with a handful
of features, usually `stdDev=0.1` works well.  However, there is an empirical rule of thumb you can use too...
If I remember correctly it is something like `stdDev = sqrt(2 / num\_features)`.


## Measuring Performance
Great performance on your training set can be meaningless: the classifier could have enough degrees of freedom that
it was allowed to memorize the training set perfectly.  This is why it is essential to have a validation set, which
allows you to track whether the network is truly learning features that generalize to the greater population.
However, as you tweak and upgrade your model based on these observations, you are indirectly encoding the aspects
of the validation set into your model.  This can run into similar problems as overtraining.  This is why it's
important to also have a test set: something you rarely pull out to test the model on.  Ideally, the test set
would only be used on the last version of your model: it will identify whether you over-validated.  

## Data Splits: Training, Validation, Test
You want to train w/ as much data as possible.  However!  You need enough data in your validation and test
sets so that the performance estimates are stable.  That is, if you only test on a few data points, it's not
clear whether the performance estimate is accurate, or if you just happened to test the model on a few data points
it by chance got right.  Moreover, it's equally unclear whether any tweaks to your model that improve or degrade the estimate 
truly improved or degraded the performance when the test/validation set is only a few data points.

## Rule of 30
Hand-wavy rule that states that an improvement/degradation of a performance estimate can be taken seriously if
it affects at least 30 data samples.  

Example \#1:  if a model has an 80% performance on a 3k-sample validation set, then any new estimate outside the 
open interval (79%,81%) can be attributed to how you tweaked the model.

Example \#2: on a 30k-sample validation set, estimate changes of 0.1% can be considered significant

CAVEAT:  The heuristic assumes that classes are balanced!  In practice, classes are almost never balanced. However,
you can treat your data in ways that induces balance (e.g., under/over sampling, SMOTE, etc).

In some treatments, your best bet is just to have more data. LOTS MORE DATA!!! MuhHHhaahaHHa!

If your data set is small in general, you can also use cross-validation.  However, as compute time currently stands,
CV is a horribly-long process on very large data sets.

## Scaling Gradient Descent
Don't use a single data point.  Don't use the full data set.  Use a mini-batch: usually between 32-1024 randomly
selected samples.  

This is stochastic gradient descent.  SGD.  Sometimes a mini-batch will point us in a bad direction, but on 
average it will point us in the right direction!  

Mini-batching has some inherent pro's that come along for the ride:
* provides the ability to train a model, even if a computer lacks the memory to store the entire dataset


## More Tricks
Zero mean and small variance is important for SGD.  There are other "tricks" that help as well.

* Momentum:  keep running average of the gradients, and use most recent average to take a step
    ```
    M <- 0.9M + gradient
    ```
* Learning Rate Decay:  How big should your learning rate be?  Good question w/o a great answer (though it should likely be in the range 0.0001-0.1). One thing that is clear: convergence to a minimum can be achieved faster if the relative size of the LR decreases over time
    - Note that large LRs are not necessarily better than small LRs; a large LR might seem to learn more quickly at first, but often comes to a plateau that a smaller LR will plow through given some time
* Adagrad: does momentum and LR decay for you
* Adam: I personally have found Adam to be the best out-of-the-box SGD enhancer


## Mini-Batching in TensorFlow
1GB of data? Meh, who cares.  10GB of data -- ok, you start getting into a range where using a GPU that can ingest
the full data set gets expensive.

But who needs an expensive GPU when you can actually just mini-batch your training process on a cheap GPU?

One caveat: when divvying up a data set into equi-sized subsets, there will likely be a "remainder subset" that does not
fully filly up to size!  e.g., batches of 3 samples from a 11-sample parent set: (1,2,3), (4,5,6), (7,8,9), (10,11)

You can choose to clip that last set, or allow the TF placeholder to take on batches of varying size (by using None as the batch size)
```python
# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])
```

EPOCH: a single forward and backward pass of the entire data set

Some Example TF Code

```python
# Assuming train_features, train_classes, etc, already defined above

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

batch_size = 128
epochs = 10
learn_rate = 0.001

train_batches = batches(batch_size, train_features, train_labels)

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch_i in range(epochs):

        # Loop over all batches
        for batch_features, batch_labels in train_batches:
            train_feed_dict = {
                features: batch_features,
                labels: batch_labels,
                learning_rate: learn_rate}
            sess.run(optimizer, feed_dict=train_feed_dict)

        # Print cost and validation accuracy of an epoch
        print_epoch_stats(epoch_i, sess, batch_features, batch_labels)

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

print('Test Accuracy: {}'.format(test_accuracy))
```

## AWS
* Remember to shut down and TERMINATE!
* ssh carnd@ipAddresss
* password protected (barmd)
* jupyter notebook
* ipAddress:8888

