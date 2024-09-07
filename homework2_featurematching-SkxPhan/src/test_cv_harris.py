# import cv2
# import numpy as np
# from skimage.feature import peak_local_max

# # path to input image specified and
# # image is loaded with imread command
# image = cv2.imread("data/Tests/one_corner.png")

# # convert the input image into
# # grayscale color space
# operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # modify the data type
# # setting to 32-bit floating point
# operatedImage = np.float32(operatedImage)

# # apply the cv2.cornerHarris method
# # to detect the corners with appropriate
# # values as input parameters
# dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
# # Or try with cv.goodFeaturesToTrack()

# # # Results are marked through the dilated corners
# # dest = cv2.dilate(dest, None)
# peak = peak_local_max(dest, min_distance=10, threshold_rel=0.05)

# # Reverting back to the original image,
# # with optimal threshold value
# image[dest > 0.1 * dest.max()] = [0, 0, 255]
# # image[peak] = [0, 0, 255]

# # the window showing output image with corners
# cv2.imshow("Image with Borders", image)

# # De-allocate any associated memory usage
# if cv2.waitKey(0) & 0xFF == 27:
#     cv2.destroyAllWindows()


import numpy as np
import cv2 as cv

img = cv.imread("data/Tests/four_corners.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray, None)

img = cv.drawKeypoints(gray, kp, img)

cv.imwrite("sift_keypoints.jpg", img)

img = cv.drawKeypoints(
    gray, kp, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
cv.imwrite("sift_keypoints.jpg", img)

# # the window showing output image with corners
cv.imshow("Image with Borders", img)

# De-allocate any associated memory usage
if cv.waitKey(0) & 0xFF == 27:
    cv.destroyAllWindows()
