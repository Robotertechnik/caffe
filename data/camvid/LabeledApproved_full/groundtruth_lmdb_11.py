import lmdb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import random

# Make sure that caffe is on the python path:
caffe_root = '../../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# color code for ground truth images
label_colors = [(64,128,64),(192,0,128),(0,128,192),(0,128,64),(128,0,0),(64,0,128),(64,0,192),(192,128,64),(192,192,128),(64,64,128),(128,0,192),(192,0,64),(128,128,64),(192,0,192),(128,64,64),(64,192,128),(64,64,0),(128,64,128),(128,128,192),(0,0,192),(192,128,128),(128,128,128),(64,128,192),(0,0,64),(0,64,64),(192,64,128),(128,128,0),(192,128,192),(64,0,64),(192,192,0),(0,0,0),(64,192,0)]
#reduced to 11 classes due GPU capacity
label_class = [255,255,0,255,1,2,255,255,3,4,255,255,255,255,255,255,5,6,255,7,8,9,255,255,255,255,10,255,255,255,255,255]

f = open('camvid_gt_TEST.txt','r')
inputs = f.read().splitlines()
f.close()

HEIGHT = 360
WIDTH = 480

# in_db = lmdb.open('camvid-gt', map_size=int(127139840)) #Height= 360 Width= 480
in_db = lmdb.open('camvid-gt_TEST', map_size=int(58720256*2.3)) #Height = 180 Width= 240
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        im = np.array(Image.open(in_).resize((WIDTH, HEIGHT), Image.NEAREST)) # downsize for reduced memory usage
        tmp = np.uint8(np.zeros(im[:,:,0:1].shape))
        for i in range(0,len(label_colors)):
            tmp[:,:,0] = tmp[:,:,0] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)

        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camvid gt TEST done'

f = open('camvid_gt_TRAIN.txt','r')
inputs = f.read().splitlines()
f.close()

# in_db = lmdb.open('camvid-gt', map_size=int(127139840)) #Height= 360 Width= 480
in_db = lmdb.open('camvid-gt_TRAIN', map_size=int(20971520*10)) #Height = 180 Width= 240
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        im = np.array(Image.open(in_).resize((WIDTH, HEIGHT), Image.NEAREST)) # downsize for reduced memory usage
        tmp = np.uint8(np.zeros(im[:,:,0:1].shape))
        for i in range(0,len(label_colors)):
            tmp[:,:,0] = tmp[:,:,0] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)

        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camvid gt TRAIN done'