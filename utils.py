'''
Image utility module
Author: Ayan Das
'''

from cv2 import warpAffine
from numpy import array, float32, vstack, empty
from numpy.random import randint

def shift_single_image(image: '(HxWxC)', dx=0, dy=0):
    'Shifts image by (dx, dy)'

    r, c, _ = image.shape # forget the channel
    M = array([[1,0,dx],[0,1,dy]]).astype(float32)

    return warpAffine(image, M, (r,c)).reshape(r*c,)

def shift_batch_rand(imbatch: '(BxH*W*C)', shape=(28,28), low=-2, high=2):
    'shifts batch of images by random amount'

    batch, _ = imbatch.shape
    r, c = shape
    imbatch = imbatch.reshape((batch,r,c,1))

    # random dx/dy
    R = randint(low, high, (batch, 2))

    # final output
    B = empty((0, r*c))

    for i in range(batch):
        B = vstack((B, shift_single_image(imbatch[i], dx=R[i][0], dy=R[i][1])))

    return B, R