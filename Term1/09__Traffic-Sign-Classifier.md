
The LeNet network we worked through in the previous module is a great start to building a traffic sign classifier.

Most of the videos in this module just review the LeNet lab.  Specifically, it covers each bit of code to make sure we understand
each part of the pipeline: data ingestion, preprocessing (reshaping, one-hot encoding, shuffling, etc), hyperparameters (e.g., 
learning rate, number of epochs, batch size), constructing the network architecture (i.e., defining the computation graph in TensorFlow),
creating data placeholders, choosing a loss function and optimizer, running a training session, validation, testing...

See my [LeNet Jupyter Notebook](https://github.com/krbnite/self-driving-car-nanodegree/blob/master/Term1/08__Lab__LeNet-in-TensorFlow/LeNet-Lab.ipynb)
for more details.

In the LeNet lab, we built the LeNet architecture and used it to classify handwritten digits in B&W images.  This ran
quickly on my laptop.  However, for traffic sign classification, we will be applying the LeNet architecture (or something
similar) to color images, which can really test one's laptop and patience.  So, for this project, we will be running
an EC2 instance on AWS w/ GPU.  

-------------------------------------------

The module also covers how to use AWS in more detail... I've written extensively about this kind of thing 
@ [krbnite.github.io](https://krbnite.github.io), so no need to waste too much time writing about it here.

-------------------------------------------

Changes to the LeNet Pipeline:
* will have to use train_test_split to split the data
* will have to accept 3-layer color images (not 1-layer B&W images)
* will have to change number of classes: instead of 10 digits, we will be classifying 43 traffic signs

Optional changes (after project submission):
* add regularization techniques, such as dropout
* can modify the layers (e.g., change width, change activation fcn, etc)
* can experiment with the network architecture/layout itself (e.g., add more layers, add an inception module!)
* tune the hyperparamters
* play with different preprocessing techniques
* augment the training data by rotating images, changing image color, etc

-----------------------------------------


The Data Set:
* http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset


