## Project Report 
### This report describes the project submission for Udacity CarND term 1 - Vehicle Detection Project.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear or RBF SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/cars-noncars.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/imag4-testexample.png
[image5]: ./examples/image5-testexample.png

[image6]: ./examples/heatmap1.png
[image7]: ./examples/heatmap2.png
[image8]: ./examples/heatmap3.png
[image9]: ./examples/heatmap4.png
[image10]: ./examples/heatmap5.png
[image12]: ./examples/heatmap6.png

[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is the project writeup_report

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook **Vehicle-Detection.ipynb, code cells 1.3-Visualize-the-sample-data-set and 1.4 Visualize HOG Features**.

I started by reading images in the dataset.  Here is example sets, one each for `vehicle` and `non-vehicle` classes objects, I viewed to get conversant with dataset:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I used random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations= 9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)` and `hog channel = 1`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters including the following parameters and setted on the indicated values below.
`cspace = 'YCrCb'
sp_size = 32
h_bins = 32
n_orient = 9
pp_cell = 8
cp_blk = 2
hog_chn = "ALL"
spatial_fea = True
hist_fea = True
hog_fea = True`

The following table describes the respective classification accuracies for various combinations of parameters.



| Num | color| Orient | pix/cell | cell/block | hog_channel | spatial | Histo | HOG | Accuracy |
|:---:|:----:|:------:|:--------:|:----------:|:-----------:|:-------:|:-----:|:---:|:--------:| 
| 1   | YUV  | 9      | 8        | 4          | Y           | Y       | Y     | Y   | 0.9637   |
| 2   | RGB  | 9      | 8        | 4          | ALL         | Y       | Y     | Y   | 0.9741   |
| 3   | HSV  | 9      | 8        | 4          | ALL         | Y       | Y     | Y   | 0.9803   |
| 4   | HLS  | 9      | 8        | 4          | ALL         | Y       | Y     | Y   | 0.9809   |
| 5   | YCrCB| 9      | 8        | 4          | ALL         | Y       | Y     | Y   | 0.9862   |
| 6   | YCrCB| 9      | 8        | 4          | ALL         | N       | Y     | Y   | 0.9662   |
| 7   | YCrCB| 9      | 8        | 4          | ALL         | N       | N     | Y   | 0.9718   |
| 8   | YCrCB| 9      | 8        | 4          | ALL         | N       | N     | Y   | 0.9718   |
| 9   | YCrCB| 12     | 8        | 4          | ALL         | Y       | Y     | Y   | 0.9840   |
| 10  | YCrCB| 16     | 8        | 4          | ALL         | Y       | Y     | Y   | 0.9834   |
| 11  | YCrCB| 9      | 8        | 2          | ALL         | Y       | Y     | Y   | 0.9865   |
| 12  | YCrCB| 12     | 8        | 2          | 1           | Y       | Y     | Y   | 0.9676   |
| 13  | YCrCB| 12     | 8        | 2          | 2           | Y       | Y     | Y   | 0.9485   |



I did the histogram bin size = 32 and spatial bin size = 32; decreasing them also resulted in more false positives.
I experimented with 'RBF'  SVM kernel and C=10.0 and got 0.995 classification accuracy; though the prediction times are 30 times slower.  I am taking help from Udacity student experience staff to reduce the prediction time.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I extracted the features, and scaled them using `StandardScaler` to properly handle different features, HOG, color and spatial. Then I passed through `GridSearchCV` to identify the `kernel` and `C` value for better accuracy.  This resulted in `kernel = 'rbf' and C = 10.0` with classification accuracy of **0.9929**.  

The relevant code is present in **Vehicle-Detection.ipynb#1.3-Visualize-the-sample-data-set, code cells 2 and 3**.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
I implemented sliding windows search with a combination of functions, **`slide_window(), in lesson_functions.py`** to get the window list with window size, overlap and start and stop locations in the image. For the project I chose 75% overlapp and search starting from ystart = 400, with 
scales ranging from 1 - 4.  Here is an example test image with sliding window search with scale = 1.  The relevant code is present in 
**Vehicle-Detection.ipynb, 2.-Finding-Vehicles - 2.1 - Implement Search Functions**.

Here is an example image.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Initially I started with 'Linear SVM' kernel and I got the classification accuracy of >= 0.98.  Then I sought help from Udacity student experience staff got the feedback that the classification accuracy should be better than 0.99.  I employed `GridSearchCV` with the parameter combination of **`parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}`**.  This resulted in **`{'kernel': 'rbf', 'C': 10}`** with classification accuracy of **0.9929**.

The following is a test example.    
![alt text][image4]
---

There are many false positives especially on left side, under the treed shadows and on the divider wall. I implemented weeding out false positives with unusually small sized vehicle detectors and in unusual locations, using 'SVM decision function confidence score in the function `find_cars()` in **Vehicle-Detection.ipynb, 3. Finding Cars**.

The following is a test example.    
![alt text][image5]

In this example, some false positive are removed, by using decision function confidence in conjunction with predicton score. However in video processing additional techniques were needed to weed out / reduce false positives, including averaging heat maps over multiple frames and ignoring vehicle detections with unusual places and unusually small sizes.


### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a https://youtu.be/BLHDzsBI1p0



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I implemented the following techniques to filter out false positives.
1. Used decision function confidence score in conjunction with svm predition score.  The logic is implemented in function `find_cars()`.
2. Implemented heatmaps and overlapping scales. The logic is presented in `find_cars()` function and heat related functions in **4.Heat Maps** section functions `add_heat()` and `apply_threshold()`.
3. Averaging the heatmaps over 'n' number of frames and then applyig threshold. The logic is implemented in function `process_image()`.
4. Ignoring detected vehicles in unusual places and with unsusually small sizes. The respective logic is implemented in `check_box_sizes()` and `check_box_location()`.  Personally I did not like this approach, though I implemented as a possibility. I would rather prefer finetuning detection algrorithms.



### Here are the frames (test images) with resulting bounding boxes and their corresponding heatmaps:

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]

![alt text][image12]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I mainly faced the false positive and accuracy issues from the implementation perspective.  

I also faced significant issue in computation capabilities of my laptop.  
Though 'rbf' kernel offered better accuracy, it took hours of computation time for predictions. I am taking help from Udacity customer experience lead. 

My pipeline will fail on generic roads such as HWY 17. I would like to use better algorithms rather than SVM.  
As with any machine learning exercise,  there are too many parameters to experiment with, color space, size of spatial and color bins, cells/block, SVM kernel and SVM parameter 'C'. The data given seems to be too generic. 

There is lot scope to sanitize data, removing time-series and picture of trees and horizon containing skies.


