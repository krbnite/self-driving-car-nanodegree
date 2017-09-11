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

