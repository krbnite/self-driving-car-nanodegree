## Number of Parameters in a Neural Network
In the multinomial logistic classifier, given a 28x28 input image and 10 target classes, 
there are 7850 parameters to learn (size of W + size of b = (28x28)x10 + 10).

That is, for any linear transformation with N inputs and K outputs, there are NK weight matrix
elements and K bias elements to learn, totaling NK+K=(N+1)K elements.


## Linear Transformations in a Nonlinear Model
Linear transformations can model targets that are linear combinations of the inputs...and that's
basically it.  If the target relies on multiplicative inputs, for example, the model might
not turn out so great.   So, in general, we do not want to use linear transformations to
model nonlinear phenomena.

That said, linear transformations are well understood, numerically stable, and have known
derivatives -- so it's convenient to house model parameters inside them.  Nonlinearities
can then be introduced to the model through "unparameterized" activation functions.  
It just so happens that when a few conditions are met, this set up can approximate any
function, so we're good mathematically as well as computationally.

The logistic model uses an activation function, but only to produce probabilities of
of targets that are essentially linearly combined inputs.  That is, the logistic model
can still only model linear relationships.  What if we want to produce a model that
can produce probabilities for input/target relationships that are more general than linear?
In this case, a nonlinearity must be introduced before the softmax at the end of the pipeline.

For example, `x -> W1 -> S -> W2 -> S` essentially outputs the probabilities of nonlinear
combinations of the inputs, S(xW1+b1)W2+b2.  However, using sigmoid or softmax activations at
hidden layers is computationally horrible (vanishing gradients galore!).  The ReLU activation
function has been found to perform best: `y = SoftMax(ReLU(xW1+b1)W2+b2)`

Like the sigmoid or softmax, the ReLU is applied component-wise over a layer vector, e.g., 
if `h1 = xW1+b1`, then `ReLU(h1) = [ReLU(h1[1]), ..., ReLU(h1[n])]`.

## ReLUs in TF
```python
import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])

# TODO: Create Model
h1 = tf.add(tf.matmul(features, weights[0]), biases[0])
a1 = tf.nn.relu(h1)
h2 = tf.add(tf.matmul(a1, weights[1]), biases[1])
init = tf.global_variables_initializer()

# TODO: Print session results
with tf.Session() as sess:
    sess.run(init)
    output = sess.run(h2)
    print(output)
```


## Chain Rule & BackProp
If one can represent a composition of functions as a graphical model, then one can also represent the
derivative of that composition as a graphical model.

<img src=./images/chain-rule.png>

This pic illustrates the chain rule graphically.  If you don't get it, stare at it a little longer.
This is great because we need it to update our model parameters, and this suggests how it can be done
in a computationally efficient manner.

<img src=./images/back-prop.png>

The "2x" in the backprop stage denotes the memory usage involved (usually backprop takes 2x what forward prop needs).


## TF Implementation: MNIST Example
In the beginning of 2017, I used a lot of TensorFlow at work when I was taking the Deep Learning nanodegree.
I remember when I finally said, "Fuck it!" and started using Keras.  The code below should indicate why (in Keras,
this would be like 5 lines of code). That said, TF is good to know b/c it is so customizable...but looking ahead in
this course, even they decided to show a little TF, then move on to the higher-level libraries, like Keras, that wrap
over TF for common, day-to-day deep learning tasks.

1. Get the data and metadata (e.g., input and output dims)
2. Add a hidden layer (and specify its width/dim)
3. Define parameters for the network (learning rate, num epochs, etc)
4. Create weight and bias tf.Variable tensors for hidden and output layers
  - make sure to explicitly name graph nodes (using name parameter) to avoid problems restoring the model later on, etc
  - see section on finetuning for more details
5. Create input and output tf.placeholder tensors
  - name 'em!
6. Add input, hidden, and output layers to graph
7. Choose a cost function and optimizer, and add them to graph
8. Create global variable initializer and model saver
9. SGD: Run a session over epochs and batches to optimize network
  - print updates throughout session
  - save model at end of session

In the below code, I mix and match from various sections... Unfortunately, the folks at
Udacity seem to change their conventions on how to name variables from segment to segment...
I tried fixing it all, but there might be an error or two in here somewhere (haven't yet run it).

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

# Input/Ouput Dimensions
n_input = 784         # MNIST data input (img shape: 28*28)
n_classes = 10        # MNIST total classes (0-9 digits)

# Add a hidden layer
n_hidden_layer = 256  # hidden layer number of features

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128  # Decrease batch size if you don't have enough memory
display_step = 1

# Store layers weight & bias
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer]), name="w1"),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]), name="w2")
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer]), name="b1"),
    'out': tf.Variable(tf.random_normal([n_classes]), name="b2")
}

# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1], name="x")
x_flat = tf.reshape(x, [-1, n_input], name="x_flat")
y = tf.placeholder("float", [None, n_classes], name="y")

# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']),\
    biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
# Output layer with linear activation
logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

# Define loss and optimizer
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)
    
# Model Performance (calculate accuracy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# Initializing the variables
init = tf.global_variables_initializer()

# Create Model Saver (to save and restore weights, etc)
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
    
        # Print status for every 10 epochs
        if epoch % 10 == 0:
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    x: mnist.validation.images,
                    y: mnist.validation.labels})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(epoch, valid_accuracy))

    # Save the model
    saver.save(sess, save_file)
    print('Trained Model Saved.')

```


Restore the model at a later point to further test:
```python
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))
```

## Go Deep, Not Wide!
In the example above, we have 2-layer neural network.  Theoretically, you can approximate
a function by making that hidden layer arbitrarily wide...but it is not the best way to do
things in practice.  Turns out, one can approximate a function much better with many fewer parameters
if one instead goes deep (parameter efficiency), e.g., if one instead uses a 3-layer NN with hidden 
layers of modest width.  It has also been found that the deep approach tends self-organize the hidden
layers into feature detectors at different levels/scales of detail (heirarchical structure), 
e.g., in image recognition, the first layer might focus on lines and edges, the second layer 
on geometrical shapes, the third layer on generic faces, etc.

## Regularization
Deep networks can have a lot of parameters, and let's face it: a common criticism of modeling in 
general is that any data set can be perfectly modeled given a model with enough parameters.

Regularization is required to set us straight.  That is, we do not necessarily want to limit
the parameter space in terms of what's possible, but we do want to limit it in terms of what
is probable.  We want the exploration space to be huge so we can find a great model, while doing
our best to eliminate the possibility of overfitting.

### Dropout
<img src=./images/dropout.jpg width=400>

Dropout is a regularization technique to reduce overfitting. The technique works by temporarily dropping neurons/nodes from the network, along with all of their associated incoming and outgoing connections. 

In TensorFlow, you can apply dropout like so:
```python
keep_prob = tf.placeholder(tf.float32) # probability to keep units
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
```

Note that keep\_prob must be treated differently during train and test time.  That is, only apply
a non-unital keep\_prob during training (e.g., 0.5).  During test time, make sure keep\_prob=1.

