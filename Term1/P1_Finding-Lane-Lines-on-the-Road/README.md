# **Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

In this project, Python and OpenCV (Open-source Computer Vision) are used to develop an analytical 
pipeline that can be used to automate lane line detection in image and movie files.  This report
reflects some lessons learned.


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My initial pipeline consisted of gray scaling, Gaussian blurring, Canny edge detection, defining an appropriate
region of interest (regional mask), detecting line segments in region of interest that met specified constraints.


Yellow or white lane lines painted on dark asphault/pavement correspond with sharp 
transitions in pixel intensity, and thus are amenable to edge detection techniques.
Since such sharp transitions in pixel intensity are preserved in a grayscaled version 
of these types of image, color can be considered of secondary importance in this task, 
and may be discarded to simplify the procedure.

Blurring the grayscaled image (using OpenCV's `GaussianBlur`) allows us the reduce 
high-frequency/noisy aspects in an image.
This is an essential step since image noise can induce spurious edge detections.
Blurring helps disambiguate the overall directionality of edges in the image, providing an
a better chance at capturing only lower-frequency contours of 
ojbects in the field of view.  A tuning parameter here is the window/kernel size: how many neighboring 
pixels should be considered when computing the blurred value (i.e., weighted average) 
at a given pixel?  The tradeoff associated with kernel size is an edge detector's sensitivity 
to image noise and its ability to localize an edge properly. 
A [5x5 kernel](https://en.wikipedia.org/wiki/Canny_edge_detector#Gaussian_filter)
is considered safe and standard, though not necessarily the right choice for all applications.  

Using OpenCV's `Canny` function, [Canny edge detection](https://en.wikipedia.org/wiki/Canny_edge_detector) 
is then applied to the blurred image to capture the broadstroke edges of objects in the image. 
Two thresholding parameters can be tuned to restrict which pixels are considered to be a part of an
edge (see [documentation](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html)).
If a pixel's gradient intensity exceeds the upper threshold, it is accepted as belonging to an edge.
If the intensity lies below the lower threshold, it is rejected.  Gradient intensities in the middle
range are accepted only if connected to a pixel exceeding the upper threshold.

At this point, we have an image with white edges sketched over a dark background.  
The question is: which edges are lane lines?  
This necessity to contextualize and properly interpret the edges is a major
step from image analysis into computer vision.  However, in our case, no black magic
(or deep learning) is (yet) necessary: contextualization can be provided by considering
the region in our images where the lane lines should appear.  

A trapezoid works!

However, more contextualization was needed when applying this pipeline to the diversity of images 
represented in the movie files.  One fix was to define an angular mask as well.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
