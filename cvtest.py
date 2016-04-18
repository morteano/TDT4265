import cv2

img = cv2.imread('stdSLR.jpg', 0)
print img.shape
equ = cv2.equalizeHist(img)
cv2.imwrite('res.png', equ)
