# Computer Vision Shape Detection
# Created by: Eduardo Trevino


#Import Required Modules
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files

#Upload and Display the Image for Shape Detection
uploaded = files.upload()
filename = next(iter(uploaded))

#read an image
img = cv2.imread(filename)

#show image
cv2_imshow(img)

# Convert shape image to greyscale (in order to make thresholding easier)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2_imshow(gray)

# Treshold the grayscale image (thresholding allows to more easily detect contours)
# setting threshold and getting a thresholded img
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# display the threshold
cv2_imshow(threshold)

# Find contours in the thresholded image drawing them in original
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
  # using draw countorus we are drawing countours
  cv2.drawContours(img, [contour], 0, (0, 165, 255), 2)

cv2_imshow(img)

# Using the contours to apporximate the shape
i = 0
for contour in contours:
  # not detecting boundary
  if i == 0:
    i = 1
    continue

  approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

  # find the center of shapes
  M = cv2.moments(contour)
  if M['m00'] != 0.0:
    x = int(M['m10']/M['m00'])
    y = int(M['m01']/M['m00'])

  if len(approx) == 3:
    cv2.putText(img, 'Triangle', (x-20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  elif len(approx) == 4:
    cv2.putText(img, 'Rectangle', (x-20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  elif len(approx) == 5:
    cv2.putText(img, 'Pentagon', (x-20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  elif len(approx) == 6:
    cv2.putText(img, 'Hexagon', (x-20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  else:
    cv2.putText(img, 'Circle', (x-20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2_imshow(img)
