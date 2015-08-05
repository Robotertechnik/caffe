import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
# init
caffe.set_mode_cpu()
#caffe.set_device(0)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('../../data/CamSeq01/0016E5_07959.png')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('../../models/bvlc_googlenet/fcn-deploy_16stride.prototxt', 'D:/camvid_experiment_03/fcn-googlenet16-bvlc_camvid_finetune_iter_7500.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
assert isinstance(net, object)
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

#print out

train_loss = np.load('D:/camvid_experiment_03/loss-googlenet16-bvlc_camvid_finetune_iter_7500.npy')

plt.subplot(1, 3, 1)
plt.imshow(im)
plt.subplot(1, 3, 2)
plt.imshow(out)
plt.subplot(1, 3, 3)
plt.plot(train_loss[0:7501])

plt.show()