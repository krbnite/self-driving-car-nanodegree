## Helping the Machine Learn
A neural network is designed to learn as best it can.  However, its best need not be impressive.
To facilitate the learning process, the network designer can take known structure into consideration.

For example, if one is attempting to classify letters, it does not matter what color a letter is to
do so, so this additional structure can be removed from the data for this task.

Another example concerns detecting an object in an image, e.g., a cat.  It does not matter if the 
cat is in the upper right-hand corner of the image, or if it's in the middle or other side.  This
represents a translational variance.  To a reasonable degree, it does not matter how big or small
the cat is in the image.  This represents a scale invariance.  One can also see that there is a
rotational invariance in this task as well.  

Similarly, in text, there is a degree of "translational invariance" for a particular word, such as 
kitten.

These latter examples are exploited using weight sharing.  For images, this issue is largely tackled
by convolutional neural networks, while for text it is approached using recurrent neural networks.

## Convolutional Neural Networks (CNNs, ConvNets)
Def: Networks that share their parameters across space.

If one takes an image, it has a width, height, and depth (e.g., RGB).  It's a 3D rectangle of input
features.  As simple neural networks go, to extract information from these features, we would first flatten
the image to a w0xh0xd0-dimensional vector, then apply a linear
transformation by applying a matrix to the inputs and adding a vector offset.  In other words, we would map
R^{w0xh0xd0} --> R^{w0h0d0} --> R^M. This would amount to w0xh0xd0xM weights, and represent unique/individual
importance to each input pixel.  

<img src=./images/flatten-image-many-weights.png width=500>

Imagine instead mapping to K M-dimensional vectors by applying to a much smaller matrix to K subsegments
of the flattened input vector.  

<img src=./images/flatten-image-fewer-weights.png width=500>

We are now using weight sharing...but perhaps not necessarily in a meaningful way.  How do the elements of each
subsegment relate to each other?  Are the pixels in each subsegment randomly selected from any location
and from any RGB channel?  Is there a way we can use weight sharing in a more structured fashion!

Yes!

Instead of flattening the full image, take K full-channel, rectangular chunks of the image (i.e., define a
window size and slide it up, down, and across the full image).  These are your
meaningful subsegments, or pixel neighborhoods.  Each pixel neighborhood gets mapped to a N-component vector, and all
the vectors form a new N-channel image.  In the new image layer, the channels no longer necessarily represent color.
Instead they represent the "window scores" of N learned features.  For example, if you're looking for cats, the 
channels would represent N features of cats that can be scored and used to infer whether a particular feature is present
in the corresponding window.  

<img src=./images/cnn.png width=400>

If you do this a few more times, you build a network of feature maps.
<img src=./images/cnn2.png width=400>

The typical architecture is to continuously shrink the spatial dimensions of the feature maps, while
increase the channel dimension.  In a sense, this is similar to Fourier analysis, where you go from
the space domain to the frequency domain.  In the current case, you go from the space domain to the
feature domain.  However, the spatial shrinking that occurs in the current case goes beyond this
analogy: the idea is that we ultimately map to the "classification domain."  That is, in many cases
we care about whether the image is classified a certain way (hasCat vs doesNotHaveCat).  However,
one might also want to localize that information...

A lot of the CNN stuff is similar in concept to Fourier time-frequency analysis.  Instead of window, 
you will see "patch" or "kernel" or "filter" (come to think of it, we use those words interchangeably
at times as well...).  Instead of step size, they use the term "stride."  What to do at the edges
is considered as well... In IDL, we had options like edge\_wrap, edge\_trunc, zero padding, etc.  In
the current case, the terminology is "valid padding" (patch does not go off image) and "same padding" (patch
will go off image onto a zero-padded region).

* http://www.matthewzeiler.com/wp-content/uploads/2017/07/eccv2014.pdf

### Padding
* Assume input of width W, height H, and depth D.
* Assume convolutional layer has a filter size of F (i.e., fw=fh=F), depth K, and stride S
* Assume zero-padding extends image P pixels from edge

General formula for new image cube:
* W\_new = 1 + (W-F+2P)/S
* H\_new = 1 + (H-F+2P)/S
* D\_new = K

Special Case: Valid Padding
* P=0
* W\_new = 1 + (W-F)/S
* H\_new = 1 + (H-F)/S
* D\_new = K

Special Case: Same Padding
* P\_W = (SW-S-W+F)/2
* P\_H = (SH-S-H+F)/2
* W\_new = W = 1 + (W-F+2P\_W)/S
* H\_new = H = 1 + (H-F+2P\_H)/S
* D\_new = K



### Summary
Slide a patch over the image.  Map each instance of the patch to an N-component vector.  This
constructures a new N-channel image.  Each channel corresponds to a feature that may or may not
be present in a given instance of a patch, but is somehow overall important to the classification
task.  For example, a pointy triangular ear is helpful in identifying a cat, and will likely show
up in a patch or two, but certainly not all the patches.  Similarly, a cat's eye will not show up
in every patch.  In reality, whether a feature is present or not is binary, but in practice, each 
feature must be scored by the network in each patch; the activation function then "binarizes" these
scores at much as possible while maintaining differentiability.  

The spatial-shrinkage architecture can help ensure that the network eventually arrives at a "yes"
or a "no".  For example, if we are looking for "cat eye", "whiskers," "pointy ear", we might not 
have a definitive "yes" or "no" in any patch as to whether or not the image has a cat in it.  However,
we can patch over this layer, forming a second hidden layer.  Again, we squeeze a pixel neighborhood
in the first hidden image into a vector of scores.  This vector might now look for a mix of features, such
 as "cat eye & whiskers", "cat eye & pointy ear", and "whiskers & pointy ear".  A final layer might score
 an even more detailed mix of input features...ultimately deciding how confident it is that a cat is in
 the original image.
 
 ### Fully-Connected Layers
 Fully-connect, or dense, layers are the regular' ol NN layers we've used since the beginning.  Usually,
 you punch a few of these layers into the network after several convolutional layers, before the
 final classification score is computed.
 
 
-----------------------------------------

Stopped at (10: Conv Output Shape)

## Padding in TensorFlow
TF does not exactly adhere to the definition of "SAME" and "VALID" padding given above.

SAME Padding, the output height and width are computed as:
* out\_height = ceil(float(in_height) / float(strides[1]))
* out\_width = ceil(float(in_width) / float(strides[2]))

VALID Padding, the output height and width are computed as:
* out\_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
* out\_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

## Number of Parameters
Q: How many parameters would a convolutional layer have if it did not use parameter sharing?
 - (H1xW1xD1 +1)\*(H2xW2xD2), where +1 is for bias

Q: How many parameters does a conv layer have w/ parameter sharing?
 - (H1xW1xD1 +1)\*20
 
 For example, imagine a 32x32x3 image is mapped to a 14x14x20 hidden image using a 8x8x20 patch.
 Without parameter sharing, this would amount to (8x8x3 +1)\*(14x14x20) = 193\*3920 = 756,560 parameters.
 However, by sharing parameters we need only worry about 3860 parameters -- less than 1\% as many parameters!
 
 ## Convolutional Layers in TF
 ```python
 # Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
```

## Explore the Design Space
After a convolutional layer, one can further downsample by using pooling techniques, such as
max pooling or average pooling.  That is, when the hidden image is formed, one can apply a "pooling
patch" over it that has its own size and stride, and computes a representative for each patch position.
For example, if we map a 32x32x3 image to a 14x14x20 image using an 8x8x20 patch, we can then use a
2x2x20 pool with a stride of 2 to churn out a 7x7x20 representation of the hidden image.

<img src=./images/max-pooling.png width=400>
<img src=./images/lenet.png width=400>

An advantage of pooling is that we can significantly reduce the parameter necessary to learn. 
However, the tradeoff is with choosing several more hyperparameters (pool size and stride).

```python
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
# Apply Max Pooling
#  -- ksize and strides take lists like [batch, height, width, channels]
#  -- most often, batch and channels are set to 1
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

Max pooling has some regularization properties to it (can help prevent overfitting).  
Some say it helps introduce a little bit of rotational invariance...

## 1x1 Convolutions
This is really just a matrix multiply on the channel dimension.  That is, if you remember that
a pxq-sized patch maps an n-channel image to an m-channel image using (p\*q\*n+1)\*m parameters,
then a 1x1-sized patch must map an n-channel image to an m-channel image using (n+1)\*m parameters.
That is, it is a linear transformation of an n-component channel vector into an m-component channel vector.

This is a way to add more nonlinearities to the CNN w/o necessarily adding too many extra parameters.

## Google's Inception Module
How do you know if you should choose a max pool, or a 1x1, 3x3, or 5x5 convolution?  
Don't choose: use 'em all!

For example, if you use SAME padding, then you can output feature maps of the same
width and height using these different convolutional techniques in parallel.  Then just
stack the outputs on top of each other.  That's what the inception module does.
If you choose each filtering scheme to have a small amount of channels, one can create
a very powerful network with relatively few parameters.

* http://nicolovaligi.com/history-inception-deep-learning-architecture.html
* https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/


## CNNs in TF
The "classic" CNN architecture (according to slide 30):  (convolutional layer, max pooling layer) xN, fully-connected layer, activation layer

Some notes:
* Post-Convolution Height and Width (Theoretical):  
  - new\_height = (input\_height - filter\_height + 2 * P)/S + 1
  - new\_width = (input\_width - filter\_width + 2 * P)/S + 1
* Post-Convolution Height and Width (VALID padding in TF):
  - out\_height = ceil(float(in\_height - filter\_height + 1) / float(strides[1]))
  - out\_width  = ceil(float(in\_width - filter\_width + 1) / float(strides[2]))
* Weight Shape for tf.nn.conv2d is [patch\_height, patch\_width, old\_depth, new\_depth]
* Bias Shape for the conv layer is [new\_depth]
* The ksize and strides input tensors for tf.nn.conv2d and tf.nn.max\_pool correspond to dims [batch, height, width, depth]
 - usually batch=depth=1 for both

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf

# Parameters
learning_rate = 0.00001
epochs = 10
batch_size = 128

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
test_valid_size = 256

# Network Parameters
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))}
    
def conv2d(x, W, b, strides=1):
    # the strides input for tf.nn.conv2d is [batch, height, width, depth]
    #  -- often batch=depth=1 for strides
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
    
def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

def conv_net(x, weights, biases, dropout):
    # Layer 1 - 28*28*1 to 14*14*32
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    # Layer 2 - 14*14*32 to 7*7*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer - 7*7*64 to 1024
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output Layer - class prediction - 1024 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Run the TF Session / Train and Validate the Network
# tf Graph input
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Model
logits = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(\
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf. global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: 1.})
            valid_acc = sess.run(accuracy, feed_dict={
                x: mnist.validation.images[:test_valid_size],
                y: mnist.validation.labels[:test_valid_size],
                keep_prob: 1.})

            print('Epoch {:>2}, Batch {:>3} -'
                  'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))

    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_valid_size],
        y: mnist.test.labels[:test_valid_size],
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))
```

## LeNet in TensorFlow
<img src=./images/lenet-architecture>

This is a Lab, the details and results of which can be found in [08\_\_Lab\_\_LeNet-in-TensorFlow](./08__Lab__Lenet-in-TensorFlow).



