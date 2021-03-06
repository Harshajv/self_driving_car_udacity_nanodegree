## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `vehicle_detection.ipynb`).
```python

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features
```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/car_notcar.png)



I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Car:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/YCrCb.png)

Noncar:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/YCrCb_noncar.png)



#### 2. Explain how you settled on your final choice of HOG parameters.

get hog feature with original color space 
![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/hog_features.png)

Channel0:
![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/hog_feature_ch1.png)

Channel1:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/hog_feature_ch2.png)

Channel2:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/hog_feature_ch3.png)


Final Feature is a combination of bin_spatial, hog_feature, color_his:



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using feature generated from the last step, concatenate car_feature and nocar_feature together.
```python

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

Using: 15 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 11988
36.73 Seconds to train SVC...
Test Accuracy of SVC =  0.9927
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

```python

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

def apply_sliding_window(image, svc, X_scaler, pix_per_cell, cell_per_block, spatial_size, hist_bins):
```


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Initial Boundingbox:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/initial_bbox.png)

Add Heatmap:

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/add_heatmap.png)

Then Label image

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/labeled_image.png)

Final detected image

![image](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/output_image/detected_image.png)









### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/Harshajv/self_driving_car_udacity_nanodegree/blob/master/Project5-Vehicle_Detection/result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

