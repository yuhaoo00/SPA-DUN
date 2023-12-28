import numpy as np
import cv2
import scipy.io as sio
from PIL import Image


img = np.float32(sio.loadmat('Dataset/Simu_test/gray/256/Aerial.mat')['orig'])
print(img.shape)
img = np.transpose(img[:16,:,:,0], (1,2,0))

img_long = img[:,:,8:].transpose((0,2,1)).reshape(256, 256*8)
Image.fromarray(np.uint8(img_long)).show()

img_diff = img[:,:,9:] - img[:,:,8:15]
img_long_diff = img_diff.transpose((0,2,1)).reshape(256, 256*7)
img_long_diff = img_long_diff/2+255./2
Image.fromarray(np.uint8(img_long_diff)).show()


mask = np.int8(sio.loadmat('Dataset/Masks/new/rand_cr50.mat')['mask'])
mask = mask[:,:,:8]

meas0 = img[:,:,:8]*mask
meas1 = img[:,:,8:]*mask
meas0 = np.sum(meas0, 2)/8
meas1 = np.sum(meas1, 2)/8


Image.fromarray(np.uint8(meas1)).show()
diff = meas1-meas0
diff = diff/2+255./2
Image.fromarray(np.uint8(diff)).show()
