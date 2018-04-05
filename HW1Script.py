import numpy as np
import cv2


# Step 1: Create a gray scale image
img = cv2.imread('HW1.jpg',0)
cv2.imshow('image',img) # image size is 1010x994 pixels
cv2.waitKey(0)
cv2.imwrite('HW1gray.jpg',img)

cv2.destroyAllWindows()
# cv2.destroyWindow('image')

# Step 2: Transformations on color image

# Blurring
img = cv2.imread('HW1.jpg')
output = cv2.blur(img, (10,10))
cv2.imshow('Blurred image', output)
cv2.waitKey(1000)
cv2.imwrite('HW1Blurred.jpg',output)

# Change the color scheme
b,g,r = cv2.split(img)
img2 = cv2.merge([g,r,b])
cv2.imshow('Color Scheme', img2)
cv2.waitKey(0)
cv2.imwrite('HW1ColorScheme.jpg', img2)

# Change it to canny image
canny = cv2.Canny(img, 50, 240)
cv2.imshow('Canny image', canny)
cv2.waitKey(1000)
cv2.imwrite('HW1Canny.jpg',canny)

# Sharpening
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
cv2.imshow('Sharpening', output_1)
cv2.waitKey(1000)
cv2.imwrite('HW1Sharpening.jpg', output_1)

# Edge Enhancement
kernel_sharpen_2 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]]) / 8.0
output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
cv2.imshow('Edge Enhancement', output_2)
cv2.waitKey(1000)
cv2.imwrite('HW1EdgeEnhancement.jpg', output_2)

# Embossing
# generating the kernels
kernel_emboss = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])

# applying the kernels to the image and adding the offset
output = cv2.filter2D(img, -1, kernel_emboss) + 128
cv2.imshow('Embossing - South West', output)
cv2.waitKey(1000)
cv2.imwrite('HW1Embossing.jpg', output)

# Step 3: Transformations on gray image

img = cv2.imread('HW1gray.jpg',0)

# Median Blur
median = cv2.medianBlur(img,5)
cv2.imshow('MedianBlur', median)
cv2.waitKey(1000)
cv2.imwrite('HW1GrayMedianBlur.jpg', median)

# Bilateral Filtering
blur = cv2.bilateralFilter(img,9,75,75)
cv2.imshow('Bilateral', blur)
cv2.waitKey(1000)
cv2.imwrite('HW1GrayBilateral.jpg', blur)

# Embossing
# generating the kernels
kernel_emboss = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])

# applying the kernels to the image and adding the offset
output = cv2.filter2D(img, -1, kernel_emboss) + 128
cv2.imshow('Embossing - South West', output)
cv2.waitKey(1000)
cv2.imwrite('HW1GrayEmbossing.jpg', output)

# Erosion
kernel = np.ones((5,5), np.uint8)

img_erosion = cv2.erode(img, kernel, iterations=1)
cv2.imshow('Erosion', img_erosion)
cv2.waitKey(0)
cv2.imwrite('HW1GrayErosion.jpg', img_erosion)

# Dilation
img_dilation = cv2.dilate(img, kernel, iterations=1)
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(1000)
cv2.imwrite('HW1GrayDilation.jpg', img_dilation)

# Enhancing Contrast
# equalize the histogram of the input image
histeq = cv2.equalizeHist(img)
cv2.imshow('Histogram equalized', histeq)
cv2.waitKey(1000)
cv2.imwrite('HW1GrayEnhancingContrast.jpg', histeq)

cv2.destroyAllWindows()
