import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Load CNN
net_full_conv = caffe.Net('bvlc_caffenet_full_conv.prototxt',
                          '../imagenet/bvlc_caffenet_full_conv.caffemodel',
                          caffe.TEST)

#net_full_conv = caffe.Net('fcn-deploy_8stride_early.prototxt',
#                          'GoogLeNet_8s_NP_DA_LB_iter_25000.caffemodel',
#                          caffe.TEST)
# load input and configure preprocessing
im = caffe.io.load_image('../images/cat.jpg')
transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
transformer.set_mean('data', np.load('../../python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
# make classification map by forward and print prediction indices at each location
out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
print out['prob'][0].argmax(axis=0)
# show net input and confidence map (probability of the top prediction at each location)
#plt.subplot(1, 2, 1)
#plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
#plt.subplot(1, 2, 2)
plt.imshow(out['prob'][0,281])

plt.show()