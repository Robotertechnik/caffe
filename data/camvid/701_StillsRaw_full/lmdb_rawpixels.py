import lmdb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

f = open('camvid_rp_TEST.txt','r')
inputs = f.read().splitlines()
f.close()

#HEIGHT = 360
#WIDTH = 480
HEIGHT = 180
WIDTH = 240
# in_db = lmdb.open('camvid-data', map_size=int(393216000)) # set height=360 widht=480
in_db = lmdb.open('camvid-test-lmdb', map_size=int(5e7)) # Height= 180 Widht= 240
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = np.array(Image.open(in_).resize((WIDTH, HEIGHT), Image.ANTIALIAS)) # downsize for reduced memory usage
        im = im[:,:,::-1]
        im = im.transpose(2,0,1)
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camvid data TEST done'

f = open('camvid_rp_TRAIN.txt','r')
inputs = f.read().splitlines()
f.close()

# in_db = lmdb.open('camvid-data', map_size=int(393216000)) # set height=360 widht=480
in_db = lmdb.open('camvid-train-lmdb', map_size=int(1e9)) # Height= 180 Widht= 240
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = np.array(Image.open(in_).resize((WIDTH, HEIGHT), Image.ANTIALIAS)) # downsize for reduced memory usage
        im = im[:,:,::-1]
        im = im.transpose(2,0,1)
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camvid data TRAIN done'