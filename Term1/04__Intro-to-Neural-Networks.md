In this module, we cover deep neural networks.  By the end of the project, we will train a DNN to drive a car in a simulator.  
This will be done by manually driving the car in the simulation to develop a training set; the DNN will then learn from the way you drive!

## Linear and Logistic Regression
Lesson starts by with classic housing prices example, showing why linear regression is useful, but also 
how a linear regression is optimized by reducing the residual error, e.g., by using the method of least 
squares (minimizing the sum of squared errors). 

This is where gradient descent is introduced: how exactly do you reduce the sum of squared errors?

Describing a linear relationship by a best fit line is not always applicable: what if your target variable
only takes on values in a binary set, e.g., {0,1} or {cat, notCat}?  This is where logistic regression
is introduced...

There are a lot of ways to paint logistic regression. A simple way is to consider it like linear regression
with a different error function to minimize.

Problem: a logistic regression computes a linear decision boundary; what if that doesn't work?

Enter neural networks.

![simple nn](./images/simple-nn.png)

Instead of figuring out one linear decision boundary, we now figure out two.
```
# Perceptrons
 |--> h1=step(<m,x>+b) --
x                        |--> y=step(ph+d)
 |--> h2=step(<n,x>+c) --
```

At first, the network does not know which lines will best partition the data, so we initialize
the weights m, n, p, and biases b, c, d randomly.  One can then compute the error in y based on actual observations,
and use gradient descent to modify the weights little by little until the error is minimized.
The trained weights represent a set of decisions that the networks has found to be most applicable
in properly answering the binary classification.

Neural networks are natural feature selectors in that a near-zero valued weight on a given input feature
at a given perceptron/node implies that the input feature is not important for the corresponding decision.
If the weights on that input are near-zero for all perceptrons/nodes in the post-input layer, then the network
has effectively realized that the feature is pretty useless in terms of the necessary decision making. 
(Note that these comments are assuming all features have been normalized; otherwise, very small or very
large weights do not necessarily measure feature importance, but how much rescaling is necessary to 
make the feature effective.)

## Building a Perceptron Network by Hand
```
# For all logic perceptrons below, x in {(0,0), (0,1), (1,0), (1,1)}
# AND Perceptron
    x --> step(x[1]+x[2]-1.1) --> {0|1}
# OR Perceptron
    x --> step(x[1]+x[2]-0.9) --> {0|1}
# NOT Perceptron (acts on 2nd component only)
    x --> step(-x[2]+1) --> {0|1}
# XOR Perceptron Network
## Might be a little crazy to write out here... 
## Just know that following the input layer, there is a 4-perceptron hidden layer, 
##    followed by a 2-perceptron hidden layer, followed by the OR gate output layer
```

## Neural Networks: C'mon, don't build them by hand!
The above logic networks were built by hand -- that is, we specified the weights 
that would mimic the function we were approximating.  In general, a neural network 
approximates a function by learning how to approximate the function.  That is, 
we do not specify the weights in a neural network because, in general, we do not
know anything about the function that is being approximated (aside from its input
and output, in the supervised case).  So, instead, a neural network is born into
the world with a bunch of nonsensical ideas (weights) on how to synthesize and understand
its observations.  However, it can learn.  Each time the net thinks it has better
figured the world out, it compares its predictions to actual observations/results, 
and notes what about its assumptions are causing any discrepancy between the two
so that it can correct for it.  This process is called gradient descent and is typically
computed via backpropagation.  Although I anthropromorphized the hell out of it, it is
is pretty basic math. 

1. "Am I wrong?"
 - i.e., choose an error function to compare outputs, y[i,j], with predictions, p[i,j], where the i indicates the data point, and j indicates which output node (if only one output node, then the notation simplifies to y[i] and p[i])
 - SSE = S[i]S[j](y[i,j] - p[i,j])^2, where the sum over i is over data points, and the sum over j is over output nodes (if there is only one output node, like in a binary classification, then this eq'n reduces to SSE = S[i](y[i]-p[i])^2)
2. "Where did I go wrong?"
 - weights in a neural network are like assumptions that help in the decision making process
 - the prediction (or "output decision") made by the network is a function of these assumptions
 - to figure out how each assumption/weight contributed to the output error, we look at the gradient of the error function w.r.t. the weights, which points in the direction in weight/bias space that will most rapidly increase the error --- we want to go in the exact opposite direction!
 - knowing the error and the gradient, we can then estimate how much to change each weight/assumption so that the network can make better "output decisions" the next time around
 
 
 http://ruder.io/optimizing-gradient-descent/index.html
