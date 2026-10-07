# https://www.geeksforgeeks.org/python/measure-size-of-an-object-using-python-opencv/
import cv2

# Load the image
path = 'input_image.png'
img = cv2.imread(path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#to separate the object from the background
ret, thresh = cv2.threshold(gray, 127, 255, 0)

# Find the contours of the object 
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
cv2.drawContours(img, contours, -1, (0,255,0), 3)

# Get the area of the object in pixels
area = cv2.contourArea(contours[0])

# Convert the area from pixels to a real-world unit of measurement (e.g. cm^2)
scale_factor = 0.1 # 1 pixel = 0.1 cm
size = area * scale_factor ** 2

# Print the size of the object
print('Size:', size)

# Display the image with the contours drawn
cv2.imwrite(path, img)
cv2.waitKey(0)

# Save the image with the contours drawn to a file
cv2.imwrite('object_with_contours.jpg', img)