# Advanced Lane Finding Project
#### Finding lane lines using computer vision

The goal of this project was to write a software pipeline to identify the lane boundaries in a video. In this writeup I'll explain how I did that.

I followed these steps to build my pipeline:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[drawCorners]: ./examples/drawCorners.png "Draw chessboard corners"
[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistort_road.png "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## How I completed the [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### Task
1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. The Writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.

#### Solution
You're reading it!

### Camera Calibration

#### Task
1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
2. OpenCV functions or other methods were used to calculate the correct camera matrix & distortion coefficients using the calibration chessboard images provided in the repository.
3. The distortion matrix should be used to undistort one of the calibration images provided as a demonstration that the calibration is correct. Example of undistorted calibration image is included in the writeup.

#### Solution
I computed the camera matrix and distortion coefficients by doing the following steps:

1. Prepare "object points" using `np.mgrid(...)`.
2. Convert image to grayscale using `cv2.cvtColor(...)`.
3. Use `cv2.findChessboardCorners(...)` to get the chessboard corners.
4. Append the obtained chessboard corners to the image points, append corresponding 3D points to object points.
5. Draw chessboard corners to check that they're correctly detected (see image below).
6. Take image points and object points to calculate distortion coefficients and camera matrix with `cv2.calibrateCamera(...)`.
7. Undistort images using `cv2.undistort(...)`. Test on chessboard images (see image below).

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`). 

I started by preparing "object points", which are the (x, y, z) coordinates of the chessboard corners in the world. I assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` were appended with a copy of it every time I successfully detected all chessboard corners in a test image.  `imgpoints` were appendeded with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Draw chessboard corners][drawCorners]
![Original and undistorted images][image1]

###Pipeline (single images)

#### Task
1. Provide an example of a distortion-corrected image.
2. Distortion correction that was calculated via camera calibration has been correctly applied to each image.
3. An example of a distortion corrected image should be included in the writeup.

#### Solution
I have combined the undistortion in a function called `cal_undistort()`. To demonstrate this step, I applied it to a road image taken from the front camera of a car (see image below).
![alt text][image2]


#### Task
2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

#### Solution
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### Task
3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

#### Solution
The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### Task
4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

#### Solution
Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### Task
5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

#### Solution
I did this in lines # through # in my code in `my_other_file.py`

#### Task
6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

#### Solution
I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

#### Task
1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

#### Solution
Here's a [link to my video result](./project_video.mp4)

---

###Discussion

#### Task
1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

#### Solution
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

