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

[//]: # (Image References)

[image1]: ./output_images/chessboard_images.jpg "Chessboard Calibration"
[image2]: ./output_images/vehicle_sign_images.jpg "Road Transformed"
[image3]: ./test_images/marks_on_road_example.jpg "Marks on Road"
[image4]: ./output_images/abs_sobel_x_only_on_red.jpg "Abs Sobel X on Red Channel"
[image5]: ./output_images/abs_sobel_and_dir_binary_on_red.jpg "Abs Sobel X AND Directional Binary on Red Channel"
[image6]: ./output_images/ch_threshold_hls_s_and_h.jpg "Ch Threshold on HLS S and H Channels"
[image7]: ./output_images/combined_binary_of_marked_road.jpg "Combined Binary of Marked Road"
[image8]: ./output_images/color_binary_of_marked_road.jpg "Color Binary on Marked Road"
[image9]: ./output_images/transform_perspective.jpg "Perspective Transform"
[image10]: ./test_images/test5.jpg "Curved Road"
[image11]: ./output_images/centroid_search.jpg "Centroid Windows Drawn on Lane"
[image12]: ./output_images/pipeline_output_example1.jpg "Pipeline Output Example 1"
[image13]: ./output_images/pipeline_output_example2.jpg "Pipeline Output Example 2"
[image14]: ./output_images/pipeline_output_example3.jpg "Pipeline Output Example 3"
[video1]: ./video_submission.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In order to undistort an image and compute the camera matrix and distortion coefficients as parameters to undistort other images, I used OpenCV's calibrateCamera function. The function requires an array of 3D coordinates that correspond to the location of points in the real world, as well as an array of 2D coordinates that correspond to the location of those points in a 2D image. For the 3D coordinates, I created an array called `objp` that is contains the x, y, z coordinates of each corner of a 9x6 square chessboard starting from (0,0,0) and the going across and down the chessboard. The z coordinate is always 0. Next, I read in 20 images of a black-white chessboard taped to a wall. These images are still distorted. Each image is an input to the OpenCV function that finds the corners within the image. If the corners are detected, then their x, y coordinates are appended to an array called `imgpoints` and a copy of `objp` is appended to `objpoints`. These arrays can be input to `cv2.calibrateCamera()` to compute the calibration and distortion matrices. I use these matrices as input to `cv2.undistort()` throughout the project in order to undistort images. I have included an example of applying the undistort function to correct an image below.

![Chessboard Calibration][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
I applied the distortion-correction to one of the images from the vehicle, where the distortion effect was obvious:
![Road Transformed][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I will demonstrate the methods I used to create a threshold binary image on the following image from the vehicle:

![Marks on Road Example][image3]

To create the threshold binary image, I used three channels: The 'Red' channel from the RGB image, the 'Saturation' channel from the HLS image, and the 'Hue' channel from the HLS image. For the Red channel, I applied an absolute Sobel gradient operation in the X direction, with a kernel of size 9 and threshold values between 30 and 100. I did a logical `and` of this gradient with a directional gradient on the Red channel, of kernel size 9 and threshold values between 0.7 and 1.3. 

The Sobel operation on the Red channel in the X direction did a good job of finding the white lane lines, however it would also pickup extraneous marks on the road. I found that if I performed a logical `and` of the output of the Sobel operation with that of the directional gradient, I could remove some of the extraneous marks. See the following for an example. The top image shows the threshold binary for a Sobel operation in the X direction on the Red channel alone. The bottom image shows the output of that operation 'anded' with the directional gradient. 

![Absolute Sobel X on Red][image4]
![Absolute Sobel X AND Directional Binary on Red][image5]

Performing a channel threshold on the HLS Saturation and Hue channels can pick up the yellow lane better, as well as fill in parts of the lanes that the operations on the Red channel did not pick up. See the image below for an example.

![Ch Threshold on HLS S and H Channels][image6]

Finally, I do a logical `or` of the Red gradient output and the HLS threshold output since both complement each other. The Red channel picks up certain white lane portions better and the HLS threshold output picks up yellow lanes better even if the lanes have shadow on them. The top image shows the combined binary output and the bottom image shows the Red channel contribution in red and the HLS channel contribution in green.

![Combined Binary][image7]
![Color Binary][image8]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I used the OpenCV functions `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` to transform an image from the vehicle to a top-down perspective. In order to do so, I had to specify the x, y coordinates of points from the original image into the x, y coordinates of those points in the transformed image. Originally, I used my code from Project 1 - Lane Finding, to find the endpoint coordinates of two lines going through both lanes, setting those as the source points, and then transforming them into fixed destination points. However, when I performed the transform, the lanes would shift around since the source points were always changing. This made it difficult for the program to keep track of the lane positions. Later, I decided to fix the source points using endpoint coordinates that roughly correspond to the points of the lane closest to the car camera and then to a position in the road about midway down the line-of-sight. This resulted in much more stable perspective transforms.

```
y_max = 450    # The y value of where the lane lines end in the distance
x_offset = 45    # The number of pixels to the left and right of the horizontal middle of the image
imshape = image.shape (720, 1280, 3)
src = np.float32([[210, imshape[0]],
                  [imshape[1]/2-x_offset, y_max],
                  [imshape[1]/2+x_offset, y_max], 
                  [1100, imshape[0]]])
dst = np.float32([[300, 720],
                  [300, 0],
                  [1000, 0],
                  [1000, 720]]) 
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 210, 720      | 320, 720      | 
| 625, 450      | 320, 0        |
| 725, 450      | 1000, 0       | 
| 1100, 720     | 1000, 720     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Transform Perspective][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use the lane-centroid search method to detect lane pixels in order to fit their positions with a 2-degree polynomial curve. If the centroid method successfully detects lane pixels and a polynomial is computed, then I use the computed coefficients to extrapolate the lane in successive frames instead of using the centroid search again.

The lane centroid search method starts by vertically summing the values in the bottom left and right quadrants of the image. It convolves the result with a 1D array of a predefined length and all the values in this array are 1. This 1D array acts as a window that can slide across the length of the image and the `argmax` of this convolution process returns the position of the lane center. The algorithm repeats this convolution process by moving up the image in predefined increments and records the left and right center lanes in an array. Once all the lane centers are found, the algorithm finds all the pixels that are within a margin surrounding each center of each lane, from the bottom to the top of the image. The image below shows a top-down view of a threshold binary image and the green windows represents the lane pixels the centroid search detected.

![Curved Road][image10]
![Centroid Windows Drawn on Lane][image11]

After I have the warped binary image and the centroid windows, I `and` them together to get an image showing only the pixels of the lanes. If the left and right lane pixels are in `left_lane` and `right_lane` respectively, then we can get the x and y values of these pixels using the following code:

```
left_lane_inds = left_lane.nonzero()
right_lane_inds = right_lane.nonzero()
# Extract left and right lane pixel positions
leftx = left_lane_inds[1]
lefty = left_lane_inds[0]
rightx = right_lane_inds[1]
righty = right_lane_inds[0]
```

The program uses the x and y values of the pixels in Numpy's `polyfit()` function to compute the coefficients of a 2-degree curve that best fits the pixel positions. I store each frame's coefficients into an array and then compute the moving average of the coefficients across the 3 most recent frames.

```
# Fit a second order polynomial to the extracted left and right lane pixels
# Set this new fit as the current fit
left_line.current_fit = np.polyfit(lefty, leftx, 2)
right_line.current_fit = np.polyfit(righty, rightx, 2)

# Add the new poly coeffs to the list of all poly coeffs
left_line.poly_coeffs.append(left_line.current_fit)
right_line.poly_coeffs.append(right_line.current_fit)

# Calculate the average poly coeffs over the last n frames
left_line.best_fit = np.average(np.array(left_line.poly_coeffs[max(0, left_line.d + 1 - n):]), axis=0)
right_line.best_fit = np.average(np.array(right_line.poly_coeffs[max(0, right_line.d + 1 - n):]), axis=0)
```

I use this "best" fit to calculate the x values of the curve that fits the two lanes, given a set of predefine y values running from 0 to 719, the height of the image.

```
# Create a 1D array from 0 to 719
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])

# Use the average poly fit coeffs for further calculations
left_fit = left_line.best_fit
right_fit = right_line.best_fit

# Use the fit coeffs to generate the x values of the curve for both lanes 
left_line.bestx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_line.bestx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I created a function called `calc_curve` to take in the x and y values of the lane pixels to calculate the radius of the curve. The code is below:

```
def calc_curve(y_eval, lefty, leftx, righty, rightx):
    '''
        Takes a set of points from the right and left lanes and then calculates the curvature
        of the curve that fits those points.
    '''

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + 
                           left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + 
                            right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad
```

In order to calculate the vehicle position with respect to the lane center, I used the x values of the left and right lane curves at the bottom of the image to calculate the average between the two. I then subtracted this number from the center of the image, which is 640. Then, I multiply it by a factor to convert the number from pixels to meters. Finally, if the number is negative, I say that the vehicle is to the left of center. Otherwise, it is to the right of center.

```
# Calculate the point midway between the lanes at the bottom of the image
midpt = (left_fitx[y_eval] + right_fitx[y_eval])/2
# Calculate the offset of the midpoint to the center of the image, in meters
offset = (midpt - 640)*xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented the pipeline in the function `run_centroid_search_pipeline()`. The pipeline undistorts an image frame from the video and creates a warped treshold binary from it. Then it uses the centroid search method to find the lane pixels. It fits a polynomial to the pixels and uses the coefficients to calculate curves that are fit onto the lanes. A green polygon is filled in between the curves. The algorithm performs an inverse perspective transform of the polygon and overlays it back onto the undistorted image of the road. The coefficients are stored and future iterations will use a moving average over the most recent 3 frames to calculate the "best" fit. If a polynomial curve is successfully generated, then the algorithm uses that fit to extrapolate the lane curves for future frames. The code checks for major shifts in the positions of the lane curves or if the lane pixels are not detected. If a frame results in major shifts or no lane pixels detected, then the previous averaged best fit is used for computing the curve. If over 3 frames consecutively contains these issues, then the code starts the centroid search over and drops all the stored fits so far.

Here are 3 pipeline output images from various stages of the video

![Pipeline Output][image12]
![Pipeline Output][image13]
![Pipeline Output][image14]
---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

