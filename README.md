**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/Selection_003.png
[image2]: ./images/Selection_004.png
[image3]: ./images/Selection_005.png
[image4]: ./images/Selection_006.png
[image5]: ./images/Selection_013.png
[image6]: ./images/Selection_007.png
[image7]: ./images/Selection_008.png
[image8]: ./images/Selection_009.png
[image9]: ./images/Selection_0131.png
[image10]: ./images/Selection_014.png
[image11]: ./images/Selection_015.png
[image12]: ./images/Selection_016.png
[image13]: ./images/Selection_017.png
[image14]: ./images/Selection_018.png
[image15]: ./images/Selection_019.png
[image16]: ./images/Selection_020.png
[image17]: ./images/Selection_021.png
[image18]: ./images/Selection_022.png


### The Dataset

The labeled [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images to train our classifier are first loaded.

Here are some of the images from the dataset.

![alt text][image1]


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

To extract hog features, hog function from the skimage library is used.
The `get_hog_features` method takes in an image along with other parameters such as cells per block, orient,pixels per cell and returns the hog features.

Here's how the HOG visualization looks like

![alt text][image3]

Next up, the `extract_features` function takes in a collection of images along with additional parameters such as colorspace and number of channels to be taken into consideration and returns set of hog features for the entire set of images.

This set is then split into training and testing set.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried with various combinations of HOG parameters and the results are shown below:

![alt text][image2]

As seen from the table:
1. Accuracy increases with the number of channels taken into consideration for extraction of HOG features.
2. RGB colorspace , expectedly, doesnt perform quite as well as other color spaces.
3. Combination of YCrCb color space, along with orient value of 10, 16 pixels per cell , and 2 cells per block , does the best job at accuracy as well as at time taken to make predictions.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Next up, in the `Train the Classifier` section, Linear SVM is instantiated and trained.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The `find_cars` function combines the job of extracting HOG features from the image and then perform subsampling, making a prediction for a particular portion of the image (window), thus implementing the sliding window approach, efficiently. Each window is defined by a scaling factor. 
The function returns the bounding boxes for regions where the prediction was true (vehicle was detected) in the image.

![alt text][image4]

However, there seems to be a problem as we explore further:
There seems to be false positives as well. 

![alt text][image9]
![alt text][image10]


##### Switching to YUV color space combination for HOG parameters resulted in lesser false positives.

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

#### Different Window Scaling factors

Various scaling factors for different window sizes were tried. Here are the results:

![alt text][image6]
![alt text][image7]
![alt text][image8]

The smallest scaling factor has been as 1.0. Anything smaller than that would result in too many false positives. Also, the regions to look for cars has been restricted for each window scale size(smaller window scales->smaller sizes->far away vehicles->smaller vertical range). Larger scale windows are reserved for near (and hence larger seeming) vehicles , thus having larger vertical region to look in.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The last part in the pipeline is the **removal of false positives**. 
Heatmap approach is used for this.

![alt text][image16]

As we can see from the original images and their heatmaps, positive detections have multiple bounding boxes, whereas false detections have just a single bounding box. We use this observation to filter out the false positives.

In the `apply threshold` function, we specify the number of bounding boxes, required to consider a detection as true. We specify 1 as our threshold.

![alt text][image17]

As we can see, false detections are successfully filtered out.

Now having left with true detections, we can label the detections using `label` function of the scipy.ndimage library.

![alt text][image18]

The final detection area is set to the extremities of each identified label:

The final result looks like this.

![alt text][image5]

#### Optimization to classifier
Following changes to the HOG parameters resulted in increased classifier accuracy :
1. Using all channels instead of one of them.
2. Changing color spaces other than RGB.
3. Varying the orient parameter value to see what gives better accuracy for each color space.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

For video implementation, one last thing has to be considered. Using information from previous frames. 
A Vehicle Detection class is created for storing bounding boxes extracted from each frame and are added to a collection.
This collective historical information is added to the heatmap and then half of this number is used as a threshold for heatmap application.

[![Project video](http://img.youtube.com/vi/r_Jc3SjHR7E/0.jpg)](http://www.youtube.com/watch?v=r_Jc3SjHR7E "Vehicle Detection Project Video.")

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Choice of classifier is a significant issue in this project. Whilst we achieve 98% accuracy , still there are false positives and in cases where the conditions are not much similar to training data, the classifier would fail.

Use of a more robust classifier/learner such as convolutional neural networks (YOLO, UNet or SSD) would perform much better.

2. The pipeline takes time in processing a particular frame and making a prediction for that frame. This is not suitable for real time situations.

Again, use of networks such as YOLO, SSD and UNet can mitigate this risk.

3. Whilst taking previous frames into consideration remove jitters from the video and to some extent remove false positives, still vehicle changing direction frequently would render this useless.
