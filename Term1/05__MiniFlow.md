# MiniFlow
The objective of this lesson is to create a "mini version" of TensorFlow.  The reason for this exercise is to help students 
better understand backpropagation and differentiable graphs so that we can appreciate how TensorFlow works under the hood.

## Graph Representations of Neural Networks
How NNs are represented has definitely confused me from time to time.  This is in part due to the fact that
they aren't always pictorialized the same.

For example, there is this type of representation:
![example](./images/example-neural-network.png)

In this picture, one representation is to put unlearnable components at the nodes (the input,
the nonlinear activation, and the output) and have the network parameters/weights defined
along the edges.  However, one might also interpret the hidden layer node as the linear-nonlinear
composition, and all edges having unit weight.

The latter representation is a little closer to how we will represent MiniFlow.
In MiniFlow, edges are unital, and each node is a mathematical operation.


## MiniFlow Nodes
In MiniFlow, we will represent all mathematical operations as nodes.  

### The Node Class
We will start by defining a node class which will be inherited by specific subclasses.


```python
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented
```

We will develop MiniFlow step by step.  So, for example, right now we only demand that
a node have a method for the forward pass.  Later on, we will include methods for the 
backward pass.

### Input Nodes

```python
class Input(Node):
    def __init__(self):
        # an Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)

    # NOTE: Input node is the only node that may
    # receive its value as an argument to forward().
    #
    # All other node implementations should calculate their
    # values from the value of previous nodes, using
    # self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        if value is not None:
            self.value = value
```

### Addition & Multiplications Nodes
To emphasize the operational nature of nodes in MiniFlow, we first create classes
for two friendly and familiar mathematical operations: addition and multiplication.
These are simple nodes -- no weights are involved, so no parameters need to be learned
or specified.

```python
#class Add(Node):
#    def __init__(self, x, y):
#        # You could access `x` and `y` in forward with
#        # self.inbound_nodes[0] (`x`) and self.inbound_nodes[1] (`y`)
#        Node.__init__(self, [x, y])
#    def forward(self):
#        x=self.inbound_nodes[0].value
#        y=self.inbound_nodes[1].value
#        self.value = x+y

class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value=0
        for node in self.inbound_nodes:
            self.value +=node.value

class Mul(Node):
    # You may need to change this...
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value=1
        for node in self.inbound_nodes:
            self.value *= node.value
```

### Linear Nodes
Now that we have a taste for defining node subclasses, let's create one of the most
important nodes in terms of creating a neural network: the linear node.

```python
class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stored within the
        # respective nodes.

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        x = self.inbound_nodes[0].value
        w = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        #self.value = b
        #for xx,ww in zip(x,w):
        #    self.value += xx*ww
        self.value = np.dot(x,w) + b
```

### Piecing together a Model
For the time being, we have enough nodes to construct a model.  We do not have enough 
machinery to learn weights and biases, but we do have enough to specify them.

But how do we piece things together?

For one, we need to specify some input nodes, then feed them into some non-input nodes, like
a linear node or two.  

%% Put a simple example here  %%

But a network is not necessarily straightforward... Which nodes need to have their output
computed first, in what order, so that subsequent nodes have all necessary inputs to compute
their outputs?

To resolve this type of issue, we need to some how sort the nodes.  The type of sort we will use
is called a topological sort.

```python
def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.
    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.
    Returns a list of sorted nodes.
    """
    input_nodes = [n for n in feed_dict.keys()]
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)
    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        if isinstance(n, Input):
            n.value = feed_dict[n]
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.
    Arguments:
        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.
    Returns the output Node's value
    """
    for n in sorted_nodes:
        n.forward()
    return output_node.value
```

Play w/ MiniFlow
```python
from miniflow import *

x, y, z = Input(), Input(), Input()

f = Add(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)
```


## Activation Functions: Why do we need them?
What we need is the capacity to model nonlinearities, which the activation functions provide.
The linear transform operations cannot, by definition, model nonlinearities in the input. A sequence
of linear transforms is still just a linear transform.  However, once we separate each linear 
transform by nonlinear activation functions, learning weights no longer reduces to learning the
best effective linear transform, but learning the best linear transform arguments to nonlinear functions.

Learning the best linear transform arguments to activation functions is another way of saying that
we are learning which features from the previous layer are important to the
decision that must be made at each node in the current layer.  

A dead node is one which outputs near zero for all data points.  Effectively, it is a node that
has to make a decision that never matters to the problem at hand...  The answer is always no.
A saturated node is one which outputs near one for all data points... Effectively, it is a node that
has to make a decision that never matters to the problem at hand...  The answer is always yes.
Both these cases are undesirable in practice and result from not being able to effectively handle
backpropagation, etc.

### Add Sigmoid Node to MiniFlow
```python
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
        
    def _sigmoid(self, x):
         return 1. / (1+np.exp(-x))   
        
    def forward(self):
        """
        Perform the sigmoid function and set the value.
        """
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)
```


## Cost Function
The cost function is a function of the network parameters -- the weights and biases.  It measures how
close the network output approximates the true output function.  A smaller cost is a better network!

The cost function is often synonymous w/ loss function.  Both names make sense:
* cost measures how much accuracy it has cost us to use the network instead of the true, likely unknown function
* loss measures how much info is lost when using the network versus the true, likely unknown function

### Add MSE node to MiniFlow
```python
class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        # Call the base class' constructor.
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        self.value = np.mean(np.square(y-a))
```

## Gradient Descent
Backpropagation. The backward pass. The process by which the network runs errors from the cost function back into previous layers, nudging weights and biases with the goal of reducing the cost function in the next forward pass.

But how should we learn from the past?  

Well, for one, we know that weights and biases numerically represent the assumptions in our decision making process. So what we need to do
is figure out how these assumptions led us astray.  That is, how much did each assumption contribute to the error computed in the cost
function?

Well, think calculus 101: the derivative of a function at a point in the function's domain measures the direction and magnitude of the function's "steepest ascent" at that point.  The parabola below illustrates this nicely!  For example, at the point to the left of the y-axis, the slope of the corresponding tangent line is negative. That is, to move in the direction of steepest ascent, walk to negative infinity on the x-axis.  Similarly, for the point to the right of the y-axis, we see that the associated line has a positive slope.  Again, this points in the direction one needs to walk to see an increase in the function f(x) = x^2.  

However, we do not want to move in the direction that increases the cost function: we want to minimize the cost function!  In calc1 terms, this means to move in the opposite direction suggested by the derivative.  This makes sense when considering the parabola: when the derivative is negative on the left-most point, we want to move positive to minimize our location on the parabola. Likewise, when the derivative is positive at the right-most point, we want to move negative to minimize our location on the parabola.

We can call this "derivative descent."

![parabola](./images/parabola.jpg)

For multivariate calculus, a generalization of the single-variable derivative is called the gradient: the gradient of a function at point in the function's domain points in the direction of the function's "steepest ascent" at that point.  Again, we just want to move in the opposite direction of that.  

For example, now consider a parabaloid, e.g., f(x,y) = a*x^2 + b*y^2.  Like the tangent lines above, the gradient's direction will always
be up along the parabaloid in the direction of "steepest ascent."  And again, we want to move in the opposite direction. So we multiply the gradient by -1 and walk in that direction:

* Df = <2ax, 2by>  # gradient points along steepest ascent
* -Df  # negative gradient points along steepest descent

But how big of a step should we take?  The gradient itself specifies relative sizes of our steps along each direction, but by itself
the gradient doesn't tell us how far we should step.  That's where the "learning rate" (or step size) comes into play.

* Starting point:  <x0, y0>
* Parabaloid Cost at that poing: a*x0^2 + b*y0^2
* Direction of steepest ascent at that point: 2<a*x0, b*y0>
* Direction of steepest descent at that point: -2<a*x0, b*y0>
* Move to this point: <x0, y0> - 2*LR*<a*x0, b*y0>

If you've normalized all your input features and randomly initialized your weights and biases, then a good learning rate
is usually in the range 1e-4 to 1e-2.

![parabaloid](./images/parabaloid.gif)

## Backpropagation
So "gradient descent" just refers to nudging weights and biases in the opposite direction of their derivative... But
how do we compute all their derivatives when some are several layers deep into the network from the cost function?

Analytically, computing all the deriviatives just involves the chain rule.  Backpropagation is the observation that
you only need to compute unique derivatives once as you travel back into the network...

Say you have this MiniFlow network:
```python
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(l2, y)
```

It is represented pictorially like:
![miniflow graph](./images/miniflow-nn-graph.jpg)

This shows how to compute the parameter gradient over the entire parameter space, (w1,b1,w2,b2) in R^4.
Notice that each unique derivative is only computed once as you go from right to left.  Parameter derivatives
that require the chain rule can be put together by multiplying unique derivatives along the path.

So, to compute the gradient, we start at the cost function and travel backward through the net:
* Let f(w1,b1,w2,b2) be the cost function
* Df = <df/dw1, df/db1, df/dw2, df/db2> 
* d1 = (df/dl2)
* <df/dw2, df/db2> = d1 \* <dl2/dw2, dl2/db2>
* d2 = dl2/ds1
* d3 = ds1/dl1
* <df/dw1, df/dw2> = d2\*d3 \* <dl1/dw1, dl1/db1>

## The Backward Pass
Now that we are getting used to how MiniFlow works, let's redefine the Node class so that we 
can fully implement self-learning neural networks.  That is, we need to (i) add a method to the Node
class for backpropagation, and (ii) generalize the function, `forward()`, that runs the model forward
to one that runs the model both forward and backward on each pass, `forward_and_backward()`.  

```python
class Node(object):
    """
    Base class for nodes in the network.

    Arguments:

        `inbound_nodes`: A list of nodes with edges into this node.
    """
    def __init__(self, inbound_nodes=[]):
        """
        Node's constructor (runs when the object is instantiated). Sets
        properties that all nodes need.
        """
        # A list of nodes with edges into this node.
        self.inbound_nodes = inbound_nodes
        # The eventual value of this node. Set by running
        # the forward() method.
        self.value = None
        # A list of nodes that this node outputs to.
        self.outbound_nodes = []
        # New property! Keys are the inputs to this node and
        # their values are the partials of this node with
        # respect to that input.
        self.gradients = {}
        # Sets this node as an outbound node for all of
        # this node's inputs.
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError
```

```python
def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()
```

We also add backward pass methods to each node subclass (see full code at end of page).

## Stochastic Gradient Descent
You can compute the cost over the entire data set, then update the weights.  This works well on
small data sets... Hell, theoretically it works well on big data sets.  However, in practice a big data
set is not amenable to vanilla gradient descent (sometimes called full gradient descent, or just gradient
descent).  

On the other extreme, for each pass one can randomly sample a single data point, compute its cost, and update the weights
accordingly.  This can be applied to large data sets because it can be parallelized.  A potential con
is that it is highly sensitive to the amount of variance in a data set and can take a long time to converge.  
This type of gradient descent is somtimes referred to a stochastic gradient descent or online gradient descent.

In the middle, you have the more general notion of "stochastic gradient descent" -- sometimes called "batch
gradient descent" or "mini-batch gradient descent" (if one consider the full data set the batch, I suppose).  
Instead of updating weights based on the full cost, or updating weights after each
single-point cost, weights are updated based on a batch cost, where the batch is a subset of the full data set.

SGD is not only computationally more efficient than single-point SGD (one can take advantage
of vectorization libraries, etc), it leads to convergence quicker and more smoothly.  

> SGD + the BackProp algorithm = standard for training neural networks

### Random Sampling
Apparently, if a data point is only used once over the k batches that make up the full data set, then 
the convergence of SGD is usually the same as GD... (Look into this more.)

So in most I've seen, when doing SGD, people split the data set up into k batches using random sampling
WITHOUT replacement.  (I wonder how doing SGD using RS w/ replacement compares...)

### Add to MiniFlow
```python
def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # TODO: update all the `trainables` with SGD
    # You can access and assign the value of a trainable with `value` attribute.
    # Example:
    for t in trainables:
        partial = t.gradients[t]
        t.value -= learning_rate * partial
```

## How to train a neural network w/ MiniFlow
```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow import *

# Load data
data = load_boston()
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10

# Randomly initialize weights and biases
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}

epochs = 10
# Total number of examples
m = X_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        forward_and_backward(graph)

        # Step 3
        sgd_update(trainables)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
```


## Some Reading
* [Vector and Tensor Derivatives (pdf)](http://cs231n.stanford.edu/vecDerivs.pdf)
* [Stochastic Gradient Descent (Wiki)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)



## Most of the MiniFlow Code

```python
"""
Implement the backward method of the Sigmoid node.
"""
import numpy as np


class Node(object):
    """
    Base class for nodes in the network.

    Arguments:

        `inbound_nodes`: A list of nodes with edges into this node.
    """
    def __init__(self, inbound_nodes=[]):
        """
        Node's constructor (runs when the object is instantiated). Sets
        properties that all nodes need.
        """
        # A list of nodes with edges into this node.
        self.inbound_nodes = inbound_nodes
        # The eventual value of this node. Set by running
        # the forward() method.
        self.value = None
        # A list of nodes that this node outputs to.
        self.outbound_nodes = []
        # New property! Keys are the inputs to this node and
        # their values are the partials of this node with
        # respect to that input.
        self.gradients = {}
        # Sets this node as an outbound node for all of
        # this node's inputs.
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError


class Input(Node):
    """
    A generic input into the network.
    """
    def __init__(self):
        # The base class constructor has to run to set all
        # the properties here.
        #
        # The most important property on an Input is value.
        # self.value is set during `topological_sort` later.
        Node.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input node has no inputs so the gradient (derivative)
        # is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1


class Linear(Node):
    """
    Represents a node that performs a linear transform.
    """
    def __init__(self, X, W, b):
        # The base class (Node) constructor. Weights and bias
        # are treated like inbound nodes.
        Node.__init__(self, [X, W, b])

    def forward(self):
        """
        Performs the math behind a linear transform.
        """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    """
    Represents a node that performs the sigmoid activation function.
    """
    def __init__(self, node):
        # The base class constructor.
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        """
        Perform the sigmoid function and set the value.
        """
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            """
            TODO: Your code goes here!

            Set the gradients property to the gradients with respect to each input.

            NOTE: See the Linear node and MSE node for examples.
            """
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] = sigmoid*(1-sigmoid)*grad_cost

class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        # Call the base class' constructor.
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) ensures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        self.m = self.inbound_nodes[0].value.shape[0]
        # Save the computed output for backward.
        self.diff = y - a
        self.value = np.mean(self.diff**2)

    def backward(self):
        """
        Calculates the gradient of the cost.

        This is the final node of the network so outbound nodes
        are not a concern.
        """
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff


def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    Arguments:

        `graph`: The result of calling `topological_sort`.
    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    # see: https://docs.python.org/2.3/whatsnew/section-slices.html
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    Arguments:

        `trainables`: A list of `Input` Nodes representing weights/biases.
        `learning_rate`: The learning rate.
    """
    # TODO: update all the `trainables` with SGD
    # You can access and assign the value of a trainable with `value` attribute.
    # Example:
    for t in trainables:
        partial = t.gradients[t]
        t.value -= learning_rate * partial
```
