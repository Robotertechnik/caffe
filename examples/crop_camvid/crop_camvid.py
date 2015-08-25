#!/usr/bin/env python
import numpy
from PIL import Image

f = open('../convert_lmdb_camvid/camvid_train.txt','r')
inputs = f.read().splitlines()
f.close()
w, h = 960, 720
left = (w - 480)/2
top = (h - 360)/2
right = (w + 480)/2
bottom = (h + 360)/2

for in_idx, in_ in enumerate (inputs):
    # load Image
    im =Image.open(in_)
    out = im.crop((left,top,right,bottom))
    out.save('C_'+in_)

print 'cropping camvid train done'

f = open('../convert_lmdb_camvid/camvid_gt_train.txt','r')
inputs = f.read().splitlines()
f.close()

for in_idx, in_ in enumerate (inputs):
    # load Image
    im =Image.open(in_)
    out = im.crop((left,top,right,bottom))
    out.save('C_'+in_)

print 'cropping camvid groundtruth train done'