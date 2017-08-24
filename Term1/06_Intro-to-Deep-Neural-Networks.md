
In this module, we learn a bit about deep learning from [Vincent Vanhoucke](http://vincent.vanhoucke.com/), a Principal Scientist 
at Google and Tech Lead in the [Google Brain](https://en.wikipedia.org/wiki/Google_Brain) team.

I won't meticulously be taking notes here since I've already gone through Vincent's [deep learning course](https://www.udacity.com/course/deep-learning--ud730)
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

