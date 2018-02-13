# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I use gaussian blur, next use canny to do edge detection
,next step is select the ROI, finally do hough transform and draw lines to get the final output.



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming is the last step of the algorithm should be optimized


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to change the last lane detection draw algorithm to improve the accuracy.



