# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of initial training set split .without is 34799
* Then I split the training data set into X_train and X_validation, the split ratio is 0.2
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is (32*32*3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data was distributed among the class.
Seeing from the bar gram we could see that there is strong imbalance among different classes. 

Seeing from the comparison between train dataset and test dataset, we could see clearly that they are almost same, this seems good, because with that we could correct evaluate our model with the default test dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


I convert the RGB image into GRAY, change the number of channel from 3 to 1
I normalized the image data so that the mean is zero and the variant is zero, just like the method which has mentioned in the udacity normalization lecture.




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Layer1: Convolutional Layer, Input:32*32*1, Output: 28*28*16, Valid Padding, With Strides=[1,1,1,1]
        Relu Activation Function
        Pooling, Input:28*28*16, Output:14*14*16, Valid Padding, With Strides=[1,2,2,1]
Layer2: Convolutional Layer,Input:14*14*16,Output:10*10*32,Valid Padding, with Strides=[1,1,1,1]
        Relu Activation Function
        Pooling, Input=10*10*32, Output=5*5*32,Valid Padding, With Strides=[1,2,2,1]
        
        flatten input=5*5*32, output=800
        
Fully connected layer1 Input=800, output=43
        Relu Activation Function
        Drop out
Fully connected Layer2 Input=400, output=43
       



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For the trainig I used Adam optimizer, which often proves to be a good choice to avoid the patient search of the right parameters for SGD, I set batch size to 128 and epoch to 20, the drop out rate is 1.0 for training.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of  0.985
* test set accuracy of 0.916

If an iterative approach was chosen:
What was the first architecture that was tried and why was it chosen?
The first architecture I have tried is exactly the LeNet example.
* What were some problems with the initial architecture?
This architecture doesn't perform drop out process, just from my point of view it is easy to lead to overfitting.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.


* Which parameters were tuned? How were they adjusted and why? Epoch,Batchsize,keep_prob,learning rate,kernel size, stride.....



