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
[Schannel]: ./examples/Schannel.png "Binary S channel examples"
[Rchannel]: ./examples/Rchannel.png "Binary R channel examples"
[image3]: ./examples/binary_combo_example.png "Binary Example"
[image4]: ./examples/warped_straight_lines.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## How I completed the [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### Task
1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. The Writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.

#### Solution
You're reading it!

---

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

---

###Pipeline (single images)

#### Task
1. Provide an example of a distortion-corrected image.
2. Distortion correction that was calculated via camera calibration has been correctly applied to each image.
3. An example of a distortion corrected image should be included in the writeup.

#### Solution
After I calibrated the camera and saved the camera matrix and distortion coefficients in global variables, I packed the image undistortion in a function called `cal_undistort()`. To demonstrate this step, I applied it to a road image taken from the front camera of a car (see image below). I apply image undistortion to all images I handle.
![alt text][image2]


#### Task
1. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image containing likely lane pixels.  
3. Provide an example of a binary image result.

#### Solution
To find out which methods I wanted to use for lane pixel identification, I implemented several functions to calculate gradients and color transforms:

1. `abs_sobel_thresh(...)` calculates the Sobel x and y and takes the absolute value of the gradient. It then applies thresholds to the image to create and return a binary image.
2. `mag_thresh(...)` calculates the Sobel x and y, then computes the magnitude of the gradient and applies thresholds to create and return a binary image.
3. `dir_threshold(...)` applies Sobel x and y, then computes the direction of the gradient and applies thresholds to create and return a binary image.
4. `hls_select(...)` transforms the given RGB image to a HLS image space and picks only the S channel. It then applies thresholds to the S-channel image to create and return a binary image.
5. `rgb_select(...)` takes the RGB image and picks only the R channel. It then applies thresholds to the R-channel image to create and return a binary image.

I looked first at the S channel and R channel binary images with different thresholds to determine if they have the desired effect of finding lane lines. You can see the results in the images below. I determined that the R channel does not provide any better or additional information to the S channel, so I dropped using it more. I determined to use S channel values between 80 and 255.

![Various S channel thresholds][Schannel]
![Various R channel thresholds][Rchannel]

Next I moved on to investigate the various gradients. I applied various kernel sizes and thresholds and determined heuristically what worked best for each gradient in x & y direction (gradx & grady), magnitude of the gradient (mag_binary) and direction of the gradient (dir_binary). I chose the following thresholds and kernels:

1. gradx: kernel size 3, keep values between 20 and 150
2. grady: kernel size 3, keep values between 20 and 255
3. mag_binary: kernel size 7, keep values between 50 and 200
4. dir_binary: kernel size 15, keep values between 0.6 and 1.3

After that, I combined the gradients in 3 different combinations to determine which one works best:

1. gradx & grady or mag
2. gradx | grady | dir & mag
3. gradx & grady | dir & mag

I determined that latter (gradx & grady | dir & mag) is the best combination. I combined my choice in a function called `filter_lanes(...)` to use it in my pipeline. You can find images corresponding to those investigations in my jupyter notebook located in `advanced_lane_finding.ipynb`.

Here are a few examples of binary images where I applied the combined thresholing. As you can see, the lanes are clearly visible - but so are also shadows that I predict will cause some trouble. But for now, that's the best I could do (and we'll see later that it works).

![Binary images][image3]

#### Task
1. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
2. An OpenCV function or other method has been used to correctly rectify each image to a "birds-eye view".
3. Transformed images should be included in the writeup.

#### Solution
The code for my perspective transform is implemented in a function called `warp(...)`. In order to perform a perspective transform, we need to pick source points (from the cameras view) and the corresponding desired points in the bird view. This can be done by taking an image where the lanes are straight, and manually choose the points from the original and in the destination image that does not exist yet. These points define the perspective transform, and are used to get the perspective transform matrix. I got the perspective transform matrix by using `cv2.getPerspectiveTransform(...)` which takes the destination and source points as parameters.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
    src = np.float32([[275, 680],
                    [595, 450],
                    [687, 450],
                    [1050, 680]])
    width = img_size[0]
    height = img_size[1]
    offset = 300
    dst = np.float32([[offset, height],
                    [offset, 0],
                    [width - offset, 0],
                    [width - offset, height]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 275, 680      | 300, 720      | 
| 595, 450      | 300, 0        |
| 687, 450      | 980, 0        |
| 1050, 680     | 980, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The latter helped tremendously to fine tune the source points such that the lines are truly parallel in birds view.

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

