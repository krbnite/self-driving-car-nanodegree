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



