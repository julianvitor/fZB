import cv2
import numpy as np
import os



vidcap = cv2.VideoCapture('video3.mp4')
count = 0
success = True
try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print('Error: Creating directory of data')




while success:
  vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*500))
  success,image = vidcap.read()

  ## Stop when last frame is identified
  image_last = cv2.imread("frame{}.png".format(count-1))
  if np.array_equal(image,image_last):
      break

  cv2.imwrite("./data/frame%d.png" % count, image)     # save frame as PNG file
  print ("Creating... frame"+str(count)+".png")
  count += 1
