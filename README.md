
## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./Pics/boxes1.PNG
[image2]: ./Pics/boxes2.PNG
[image3]: ./Pics/boxes3.PNG
[image4]: ./Pics/raw_img.PNG
[image5]: ./Pics/heatmap.PNG
[image6]: ./Pics/ensemble1.PNG
[image7]: ./Pics/ensemble2.PNG
[image8]: ./Pics/ensemble3.PNG
[image9]: ./Pics/centerpts1.PNG
[image10]: ./Pics/centerpts2.PNG
[video1]: ./solution_project.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Main Idea
Instead of using HOG features I wanted to try other (less hard-coded) ideas which also was encouraged in the lecture videos. My idea was to use a convolutional neural net to classify cars and combine that with a sliding window search. To eliminate false positives, I used a belief and a second neural net checkup. The second checkup was also used to refine the borders of the boxes.

### Classification of Cars

#### Data Collection
I used data sets mentioned in the lecture. In total, there were 8792 car and 8968 non-car images of shape (64,64,3).

#### The Neural Net
The convolutional neural net I constructed has the following structure:

Layer index   | Layer type    | Output shape  |
------------- | ------------- | ------------- |
0 | Input         | (64, 64, 3)   |
1 | Normalization(1) | (64, 64, 3)   |
2 | MaxPooling(2) | (32, 32, 3)   |
3 | 3x3 Conv(3)   | (32, 32, 20)   |
4 | 1x1 Conv(3)   | (32, 32, 5)   |
5 | 5x5 Conv(3)   | (32, 32, 2)   |
6 | Merge(3,4,5)  | (32, 32, 27)   |
7 | MaxPooling    | (16, 16, 27)   |
8 | BatchNorm     | (16, 16, 27)   |
9 | 3x3 Conv      | (16, 16, 10)   |
10| MaxPooling    | (8, 8, 10)   |
11| BatchNorm     | (8,8,10) |
12| Flatten       | (640) |
13| Dense         | (50) |
14| Dropout       | (50) |
15| Dense         | (20) |
16| Dropout       | (20) |
17| Dense         | (15) |
18| Dense         | (1)  |

In total, the net has 36.721 parameters.

#### Training the Net

I trained the net on the AWS server on a g2.2xlarge instance. Here, one epoch took about 3 seconds of training and I trained the net for 40 epochs with a train/validation split of 0.8/0.2.

The validation accuracy was 99.58% and the training accuracy was 99.75%. Thus, the neural net proves to be a good classifier for the task of recognizing cars.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the sliding window search as presented in the lecture. I used the following eight windows for each frame:

```python
sizes = [(90,50,500), (75,75,500), (150,75,500), (100,100,550), (200,100,600), (125,125,600),(150,150,660), (200,200,660)]
```
Here, the first number is the size in x-direction, the second the size in y-direction in the third number represents the part of the image where this window is used in y-direction, e.g., we use the part
```python
img[350:500, :]
```
for the window (90,50,500). We always start at 350 to delete the sky. In total, with an overlap of 0.6 there are 342 windows in each frame that are checked by the neural net. This whole process took approximately 1.5 seconds on the AWS server.

Since we were only interested in the cars on our lane, I started the sliding window search at 600 pixels in x-direction.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The following pictures show the results of the sliding window searches for some of the test images:

![alt text][image1]

![alt text][image2]

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./solution_project.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I constructed a heatmap using the sliding window search combined with the CNN classifier. I summed up the heatmap over 5 frames and thresholded each incoming heatmap as well as the summed up heatmap. Then, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. The next pictures show the raw image, the heatmap and the final boxes for the image:

![alt text][image4]

![alt text][image5]

![alt text][image1]

There were still some problems to tackle:
* Combining overlapping boxes
* Refining the box borders such that the car is inside the box
* Minimizing false positives
* Smoothing the result

#### 3. Minimizing False Positives and Refining
The main technique I used was belief. The center of each detected box was safed as long as there were windows that covered this point. If a point was covered for 7 frames, a box that covered this point was accepted. For each remaining box, I used the following procedure:

I created 100 scenarios of boxes similar to the box. Each of these boxes was fed to four CNNs (they were similar to the one described above). If three of the CNNs classified an input as a car, the box was accepted. In the end, I took the largest accepted box out of these 100 scenarios. This should also create boxes that cover the whole car. On the flipside, some boxes are therefore too large.

The next pictures show the box from the sliding window search, then the box that was the output by the neural network ensemble and then the final box after averaging with the history. We can observe that the history output actually seems to be worse but in a continuous stream it still works better. Here, the box is shifted to the right because the car came from the right.

![alt text][image6]

![alt text][image7]

![alt text][image8]



#### 4. Combining Overlapping Boxes
If there  both upper left and lower right corners were different in less than 20%, I averaged these boxes to one box.


#### 5. Smoothing the Result
In order to smooth the result, I saved the boxes for 15 frames and averaged similar to the overlapping step. In the following image, we can see a series of 'still living' center points (red) and the feasible bounding boxes around them. Below is the final result after ensemble checking and smoothing.

![alt text][image9]

![alt text][image10]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Although my classifier has a great accuracy, I still faced the problem with false positives. Maybe, more data would help here. Also one could replace the sliding window technique with a neural net that directly proposes a bounding box (if corresponding data was available).

The major problem of my pipeline occurs when the two cars overlap. Ideally, the pipeline should remember that a car is covered by another car. Also, one could improve the size of the bounding box to avoid too large or too small boxes. In particular, since my boxes are large, there is only one large bounding box when the cars are close to each other.
