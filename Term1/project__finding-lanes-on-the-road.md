Goal:  detect and track lane lines


## Color Thresholding
The Idea: Lanes on the highway are white, so let's remove (black out) any pixel that's not white.
The Implementation:  A white pixel is (255, 255, 255) in RGB space, so one might first assume to blackout 
any pixel with an R, G, or B value <= 254.  However, this is too strong a threshold and results in 
a completely blacked out image.  Ultimately, setting the color thresholds somewhere around 200
resulted in the type of image we imagined.

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('test.jpg')

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)

# Define color selection criteria
###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
red_threshold = 199
green_threshold = 199
blue_threshold = 199
######

rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Do a boolean or with the "|" character to identify
# pixels below the thresholds
thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]

# Display the image                 
plt.imshow(color_select)

# Uncomment the following code if you are running the code locally and wish to save the image
# mpimg.imsave("test-after.jpg", color_select)
```

## Region Masking
The black-and-white image retains the lane lines, as well as some undesirables.  
We do not want to fool our lane detection scheme: it should not interpret non-lanes as lanes!

To strengthen our technique, we can supplement our color masking technique with 
region masking.  

Note that our technique need not analyze an arbitrary image.  Instead, given the camera
is mounted on the dashboard, we need only analyze images from a known perspective.  
In this perspective, the lanes bounding our car take up a triangular region.  
We can use this information to make our lane detection scheme more robust.

We can do better than triangular masking: while driving, staying inside a line is most
sensitive to the lane lines in nearby region, which can be bounded by a quadrilateral.  

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print some stats
image = mpimg.imread('test.jpg')
print('This image is: ', type(image), 
         'with dimensions:', image.shape)

# Pull out the x and y sizes and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
region_select = np.copy(image)

# Define a triangle region of interest 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# Note: if you run this code, you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz 
left_bottom = [0, 539]
right_bottom = [900, 300]
apex = [400, 0]

# Fit lines (y=Ax+B) to identify the  3 sided region of interest
# np.polyfit() returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Color pixels red which are inside the region of interest
region_select[region_thresholds] = [255, 0, 0]

# Display the image
plt.imshow(region_select)
```



## Altogether now

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('test.jpg')

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)
line_image = np.copy(image)

# Define color selection criteria
# MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
red_threshold = 200
green_threshold = 200
blue_threshold = 200

rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define the vertices of a triangular mask.
# Keep in mind the origin (x=0, y=0) is in the upper left
# MODIFY THESE VALUES TO ISOLATE THE REGION 
# WHERE THE LANE LINES ARE IN THE IMAGE
left_bottom = [130, 539]
right_bottom = [820, 539]
apex = [450, 310]

# Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
# np.polyfit returns the coefficients [A, B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
                    
# Mask color and region selection
color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
# Color pixels red where both color and region selections met
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

# Display the image and show region and color selections
plt.imshow(image)
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
plt.plot(x, y, 'b--', lw=4)
plt.imshow(color_select)
plt.imshow(line_image)
```

--------------------------------------------------------------------------

# Computer Vision
Ok, so we figured out how to detect lane lines....that are white...during sunny daylight conditions.

Uh-oh. Better not put our self-driving car on the road quite yet.  We need to be a bit more sophisticated!

Note: at this point in the nanodegree, they recommend supplementing the material with Udacity's
[Intro to Computer Vision](https://www.udacity.com/course/introduction-to-computer-vision--ud810) course.

We will be using [OpenCV](http://opencv.org/), an open-source computer vision library.

## Canny Edge Detection
* developed in 1986

The goal of edge detection is to identify the boundaries of an object in an image.
One way to do that is to first convert the image to grayscale, compute the gradient of
the grayscale image, and identify edges by tracing along pixels with the strongest
gradients (the brightness of each pixel 
corresponds the the strength of the gradient at that pixel, and strong gradients typically
correspond to edges of an object).

```python
edges = cv2.Canny(grayImg, low_threshold, high_threshold)
```

```
# Do all the relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in the image and convert to grayscale
# Note: in the previous example we were reading a .jpg 
# Here we read a .png and convert to 0,255 bytescale
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Define a kernel size for Gaussian smoothing / blurring
kernel_size = 3 # Must be an odd number (3, 5, 7...)
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# Define our parameters for Canny and run it
low_threshold = 100
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')
```

## The Hough Transform
A line in 2D Euclidean space can be represented as y = m*x + b. In "Hough space," the same line
can be represented as a single point, (m,b).

As a set of points, a line in Euclidean space is L(m,b) = {(x,y): y = m*x + b}.  The Hough transform
reduces the number of points needed in the line set to one:  L = {(m,b)}.  

Hough Transform:  b = (-x)*m + y

A single point in 2D Euclidean space is a line in Hough space:
b = (-x0)*m + y0  (any point (b,m) that satisfies this equation).




### More on the Hough Transform
* 2004: van Ginkel et al: [A short introduction to the Radon and Hough transforms and how they relate to each other](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.2.9419)
* 2009: Hart: [How the Hough Transform was invented](https://scholar.google.com/scholar?hl=en&q=Hart%2C+P.+E.%2C+%22How+the+Hough+Transform+was+Invented%22&btnG=&as_sdt=1%2C31&as_sdtp=)


