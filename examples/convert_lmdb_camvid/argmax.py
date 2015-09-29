caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from PIL import Image
import scipy

f = open('../camvid_groundtruth.txt','r')
inputs = f.read().splitlines()
f.close()

net = caffe.Net('../googlenet_12outputs/late_deploy.prototxt', '../googlenet_12outputs/googlenet_8_12.caffemodel', caffe.TEST)

for in_idx, in_ in enumerate(inputs):
    im = Image.open('../../data/camvid/701_StillsRaw_full/'+in_+'.png')
    im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.ANTIALIAS)
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
    scipy.misc.imsave('small_inputs/argmax'+str(in_idx+1).zfill(5)+'.pgm' , out)

print 'camvid done'



