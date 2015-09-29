caffe_root = '../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from PIL import Image
import os
import scipy

path1 =  '../data/camvid/701_StillsRaw_full/'
path2 = '../data/camvid/argmax/'
# load net
net = caffe.Net('Googlenet_8s/deploy.prototxt', 'Googlenet_8s/snapshot/train_iter_80000.caffemodel', caffe.TEST)

listing = os.listdir(path1)
for file in listing:
    im = np.array(Image.open(path1 + file))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    # save image
    thisFile = file
    base = os.path.splitext(thisFile)[0]
    scipy.misc.imsave(path2+'argmax'+base+'.pgm' , out)

print 'camvid done'

#path1 =  '../../Desktop/Dusk_test/'
#path2 = '../../Desktop/argmax_seq05/'
#
#listing = os.listdir(path1)
#for file in listing:
#    im = np.array(Image.open(path1 + file))
#    in_ = np.array(im, dtype=np.float32)
#    in_ = in_[:,:,::-1]
#    in_ -= np.array((104.00698793,116.66876762,122.67891434))
#    in_ = in_.transpose((2,0,1))
#    # load net
#    net = caffe.Net('Alexnet_8s/deploy.prototxt', 'Alexnet_8s/alexnet8s.caffemodel', caffe.TEST)
#    # shape for input (data blob is N x C x H x W), set data
#    net.blobs['data'].reshape(1, *in_.shape)
#    net.blobs['data'].data[...] = in_
#    # run net and take argmax for prediction
#    net.forward()
#    out = net.blobs['score'].data[0].argmax(axis=0)
#    # save image
#    thisFile = file
#    base = os.path.splitext(thisFile)[0]
#    scipy.misc.imsave(path2+'argmax'+base+'.pgm' , out)
#
#print 'dusk test sequence done'

