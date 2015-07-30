# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = np.array(Image.open('../../data/CamSeq01/0016E5_08037.png').resize((240, 180), Image.ANTIALIAS))
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('deploy2.prototxt', 'train_iter_200.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

#print out
plt.subplot(1, 2, 1)
plt.imshow(im)
plt.subplot(1, 2, 2)
#plt.imshow(in_)
plt.subplot(1, 2, 2)
plt.imshow(out)
plt.show()
