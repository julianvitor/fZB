import cv2
import numpy as np

alpha = float(3)     # Simple contrast control 1-3
beta = int(100)             # Simple brightness control 0 -100
 
print (" Basic Linear Transforms ")
print ("-----------------------------")

img = cv2.imread('imagem4.jpg')
 
mul_img = cv2.multiply(img, np.array([alpha]))                    # mul_img = img*alpha
new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta
cv2.imwrite('contraste.png', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
