## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `camera_calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/output_images/find_corners.png)



![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/output_images/after_undistortion.png)
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image ( `binarization_mask_calculation.py`).  
Setp 1: I calculate all the four different sobel mask from the lecture, Gradx,Grady,mag_gradient, dir_gradient, Here I visualize the four different gradient threshold.
![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/output_images/sobel_threshhold_comparison.png)

Step 2: Color mask: I change the original color space RGB into HSL. Extract yellow lines and white lines respectively


Here is the example output of yellow_mask:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/output_images/yellow_mask.png)

and here is the example output of white_mask:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/output_images/white_mask.png)

Output of final combined mask:(I use only gradx for the computation of final mask)

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/output_images/final_mask_for_test5.png)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspect_trans`, which appears in the file `perspective_transformation.py`  The `perspect_trans()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python

    x1=205
    x2=1100
    y_rough=460
    offset= 100
    
    
    src=((x1,img.shape[0]),(x2,img.shape[0]),(700,y_rough),(585,y_rough))
    
    dst=((x1+offset,img.shape[0]),(x2-offset,img.shape[0]),(x2-offset,0),(x1+offset,0))
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 700, 460      | 305, 0        | 
| 205, 720      | 305, 720      |
| 1100, 720     | 1000, 720      |
| 585, 460      | 1000, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/output_images/BirdEye_comparison.png)

and after binarization:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/output_images/birdeye_binary_comparison.png)

and the curve lines perspective transformation results for detection:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/output_images/birdeye_for_line_detection.png)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

I define the Line() class in `lane_detection.py`

```python

class Line:
    """
    Class to model a lane-line.
    """
    def __init__(self, buffer_len=10):

        # flag to mark if the line was detected the last iteration
        self.detected = False

     ...
     ...
     ...
     


```

My method is for the first frame for detected, use the sliding window to detect the lane lines, and there is an example for output of sliding_windows detection

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/output_images/sliding_windows_example_output.png)

and for other frames the detection should be based on the first frame, as introduced in lecture.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `lane_detection.py` and `image_processing_pipeline.py`

```python

   def curvature(self):
        y_eval = 0
        coeffs = self.average_fit
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    # radius of curvature of the line (averaged)
    def curvature_meter(self):
        y_eval = 0
        coeffs = np.mean(self.recent_fits_meter, axis=0)
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])
```


```python

def compute_offset_from_center(line_lt, line_rt, frame_width):
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1
    
    return offset_meter

```



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project4-Advanced_Lane_Finding/out_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
