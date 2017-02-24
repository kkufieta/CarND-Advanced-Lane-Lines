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
[binaryBirdView]: ./examples/binaryBirdView.png "Binary Bird View"
[image5]: ./examples/color_fit_lines.png "Fit Visual"
[image6]: ./examples/example_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## How I completed the [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### Task
1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. The Writeup / README should include a statement and supporting figures / images that explain how each rubric item was addressed, and specifically where in the code each step was handled.

#### Solution
You're reading it!

Note: All code lines I refer to from now on are found in file `examples/advanced_lane_finding.py`

---

### Camera Calibration

#### Task: Camera matrix and distortion coefficients
1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
2. OpenCV functions or other methods were used to calculate the correct camera matrix & distortion coefficients using the calibration chessboard images provided in the repository.
3. The distortion matrix should be used to undistort one of the calibration images provided as a demonstration that the calibration is correct. Example of undistorted calibration image is included in the writeup.

#### Solution
I computed the camera matrix and distortion coefficients by doing the following steps:

1. Prepare "object points" using `np.mgrid(...)` (Lines 51 - 56).
2. Convert image to grayscale using `cv2.cvtColor(...)` (Line 67).
3. Use `cv2.findChessboardCorners(...)` to get the chessboard corners (Line 72).
4. Append the obtained chessboard corners to the image points, append corresponding 3D points to object points (Lines 79 - 82).
5. Draw chessboard corners to check that they're correctly detected (see image below) (Lines 84 - 86).
6. Take image points and object points to calculate distortion coefficients and camera matrix with `cv2.calibrateCamera(...)`  (Lines 99 - 100).
7. Undistort images using `cv2.undistort(...)` (Line 103). Test on chessboard images (see image below) (Lines 106 - 128).

I started by preparing "object points", which are the (x, y, z) coordinates of the chessboard corners in the world. I assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` were appended with a copy of it every time I successfully detected all chessboard corners in a test image.  `imgpoints` were appendeded with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Draw chessboard corners][drawCorners]
![Original and undistorted images][image1]

---

###Pipeline (single images)

#### Task: Distortion-corrected image
1. Provide an example of a distortion-corrected image.
2. Distortion correction that was calculated via camera calibration has been correctly applied to each image.
3. An example of a distortion corrected image should be included in the writeup.

#### Solution
After I calibrated the camera and saved the camera matrix and distortion coefficients in global variables (Lines 99 - 100), I packed the image undistortion in a function called `cal_undistort()` (Lines 102 - 104). To demonstrate this step, I applied it to a road image taken from the front camera of a car (see image below) (Lines 130 - 140). I apply image undistortion to all images I handle.
![alt text][image2]


#### Task: Use of color transforms & gradients to generate thresholded binary images
1. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image containing likely lane pixels.  
3. Provide an example of a binary image result.

#### Solution
To find out which methods I wanted to use for lane pixel identification, I implemented several functions to calculate gradients and color transforms:

1. `abs_sobel_thresh(...)` calculates the Sobel x and y and takes the absolute value of the gradient. It then applies thresholds to the image to create and return a binary image (Lines 154 - 177).
2. `mag_thresh(...)` calculates the Sobel x and y, then computes the magnitude of the gradient and applies thresholds to create and return a binary image (Lines 179 - 197).
3. `dir_threshold(...)` applies Sobel x and y, then computes the direction of the gradient and applies thresholds to create and return a binary image (Lines 199 - 219).
4. `hls_select(...)` transforms the given RGB image to a HLS image space and picks only the S channel. It then applies thresholds to the S-channel image to create and return a binary image (Lines 224 - 234).
5. `rgb_select(...)` takes the RGB image and picks only the R channel. It then applies thresholds to the R-channel image to create and return a binary image (Lines 236 - 243).

I looked first at the S channel and R channel binary images with different thresholds to determine if they have the desired effect of finding lane lines (Lines 250 - 290). You can see the results in the images below. I determined that the R channel does not provide any better or additional information to the S channel, so I dropped using it more. I determined to use S channel values between 80 and 255.

![Various S channel thresholds][Schannel]
![Various R channel thresholds][Rchannel]

Next I moved on to investigate the various gradients (Lines 298 - 391). I applied various kernel sizes and thresholds and determined heuristically what worked best for each gradient in x & y direction (gradx & grady), magnitude of the gradient (mag_binary) and direction of the gradient (dir_binary). I chose the following thresholds and kernels:

1. gradx: kernel size 3, keep values between 20 and 150
2. grady: kernel size 3, keep values between 20 and 255
3. mag_binary: kernel size 7, keep values between 50 and 200
4. dir_binary: kernel size 15, keep values between 0.6 and 1.3

After that, I combined the gradients in 3 different combinations to determine which one works best:

1. gradx & grady | mag
2. gradx | grady | dir & mag
3. gradx & grady | dir & mag

I determined that latter (gradx & grady | dir & mag) is the best combination. I combined my choice in a function called `filter_lanes(...)` (Lines 405 - 426) to use it in my pipeline. You can find images corresponding to those investigations in my jupyter notebook located in `advanced_lane_finding.ipynb`.

Here are a few examples of binary images where I applied the combined thresholding. As you can see, the lanes are clearly visible - but so are also shadows that I predict will cause some trouble. But for now, that's the best I could do (and we'll see later that it works).

![Binary images][image3]

#### Task: Perspective transform
1. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
2. An OpenCV function or other method has been used to correctly rectify each image to a "birds-eye view".
3. Transformed images should be included in the writeup.

#### Solution
The code for my perspective transform is implemented in a function called `warp(...)` (Lines 469 - 501). In order to perform a perspective transform, we need to pick source points (from the cameras view) and the corresponding desired points in the bird view. This can be done by taking an image where the lanes are straight, and manually choose the points from the original and in the destination image that does not exist yet. I chose to hardcode the source and destination points in the following manner (Lines 475 - 485):

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


These points define the perspective transform, and are used to get the perspective transform matrix. I got the perspective transform matrix by using `cv2.getPerspectiveTransform(...)`  (Lines 490 - 493) which takes the destination and source points as parameters.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The latter helped tremendously to fine tune the source points such that the lines are truly parallel in birds view.

![alt text][image4]

#### Task: Identify lane-line pixels & fit a polynomial
1. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
2. Methods have been used to identify lane pixles in the rectified binary image.
3. The left & right line have been identified and fit with a curved functional form (e.g. spine or polynomial).
4. Include example images with line pixels identified and a fit overplotted.

#### Solution
Once I finished the filter (getting binary images) and the transformation transform, I had binary images of the lane lines in bird's view (see image below). 

![alt text][binaryBirdView]

Using those, I implemented line detection based on a histogram and sliding windows. I did that in the function called `detect_lines(...)` (Lines 568 - 672) which takes the warped image in bird's view as an input. The histogram in the function will detect where the lines clearly stand out (Line 570). By splitting the image in two (we assume the car is between the lines, we assume that the lane lines are where the highest peak is in each of the halves (Lines 575 - 579). We save those values as `leftx_base` and `rightx_base`. Next, we choose a number of sliding windows and their size, and we set their current position based on where we found the left & right lanes from our histogram. Next, we take all the nonzero points within the sliding window and attach them to the lists of values for the left & right lane. If we have enough pixels identified in the current region (Lines 614 - 617), we adjust the x position of the sliding window based on where the mean of the nonzero pixels is (Lines 620 - 624), and move the sliding windows up  (Line 603). Once we searched that way through the entire image, we move on to fit a second order polynomial function to the pixels found for the left and right lane with `np.polyfit(...)` (Lines 636 - 638). I save the results in the variables `left_fit` and `right_fit`.

You can see two examples of the sliding windows in action in the image below.

![alt text][image5]

Note: I used the code from the lecture here, this was very advanced and I would not have been able to come up with this myself. Hopefully soon though, I'm learning!

#### Task: Calculate radius of curvature and relative position of vehicle
1. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

#### Solution

I calculated the radius of curvature and the relative position of the vehicle in `detect_lines(...)` (Lines 643 - 670), right after I fitted polynomials to the left and right lanes. I used a conversion from pixel space to meters like this (Lines 643 - 647):

```
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

I fit new polynomials to x and y in world space, using the converted points. Next, I calcualted the radius of curvature based on the equation given in the lecture.

To get the relative car position, I calculated the left and right lane positions at the height of the car and used those to get the middle of the lane in pixel values. I determined the distance of the car relative to the middle of the lane, and converted that to meters using the conversion mentioned above.

#### Task: Plot results back down on lane area
6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

#### Solution
I tested my pipeline `detect_lines(...)` on provided test images (Lines 677 - 692). I provided results for two example images to compare the values. As you can see, the radius of curvature is reasonable: For an almost straight road it is estimated as a 1 km radius, and for a slight turn left, it is estimated as 337 m, which seems to make sense. In the lecture it was mentioned that the first left curve in the project video has a radius of 1 km. My estimates vary between 400 - 500 m, so I am probably off by a factor of 2, which isn't too bad for a start.

![alt text][image6]

---

###Pipeline (video)

#### Task: Apply pipeline to video
1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

#### Solution
I applied the pipeline to all three videos (Lines 762 - 780).
Here's a [link to my video result](processed_video.mp4). 

I did the challenge videos for fun and out of curiousity, but it didn't go well. The project video works reasonably well, there are only a few outliers when there is a lot of shadow. By adding a filter I could solve that. It's hard to predict what the car would do, but at that speed the lanes would probably be detected well again before the car would go off the street.

---

###Discussion

#### Task: Drawbacks of Pipeline, ToDos for the future
1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

#### Solution
I implemented a working pipeline to detect lane lines in certain conditions. The pipeline works well for the project video, but the conditions are close to perfect in that video. It fails only rarely when the shadows are large, and that could be avoided if I implemented a simple filter to throw out outliers (badly detected lines). 

My pipeline does not work for the challenge videos, because there are too many other "lines" that it detects as lane lines. I might be able to solve that for the easier challenge video by cropping the sight area of the car (like we did in Project 1), but that would not work for the harder challenge video (probably?). I'd love to learn how those challenge videos can be solved.

Here's what I'd like to do to improve my pipeline: Check my detected lines for sanity. Make sure they're parallel, that the calculated radius and vehicle position is only slightly deviating from the previous one. That way I can tell if I truly detected lane lines, or if I failed at detecting lines. 

I would also like to play more with the parameters and various ways to detect channels. I did not like that I didn't filter out shadows. If I would manage to do that, I'd have one less problem.

Furthermore I'd like to try different solutions to detect and calculate the lines. I would have to do some literature research, and see if I can find other interesting solutions to the problem.

My pipeline will also not work in real time, since I'm recalculating sliding windows on each image, I do a lot of unnecessary computations. I would like to implement a faster solution where I can use the previously detect lanes to make the next lane computation easier. That should also add some more robustness (since I'm searching in the vicinity to the previous lane lines). 

