
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/fcn
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import lmdb
from PIL import Image
from PIL import ImageOps
import numpy as np

w, h = 960, 720
left = (w - 480)/2
top = (h - 360)/2
right = (w + 480)/2
bottom = (h + 360)/2

ridx = np.arange(367)
np.random.shuffle(ridx)

f = open('camvid_train.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camvid_train-augmented-lmdb', map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.ANTIALIAS) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

#Cropping
f = open('camvid_train.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camvid_train-augmented-lmdb', map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = Image.open(in_) # load image
        im = im.crop((left,top,right,bottom))
        im = np.array(im) # convert to nparray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

#Mirroring
f = open('camvid_train.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camvid_train-augmented-lmdb', map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = Image.open(in_) # load image
        im = ImageOps.mirror(im)
        im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.ANTIALIAS) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

#Cropping & Mirroring
f = open('camvid_train.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camvid_train-augmented-lmdb', map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = Image.open(in_) # load image
        im = im.crop((left,top,right,bottom))
        im = ImageOps.mirror(im)
        im = np.array(im) # convert to nparray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camvid train done'

# color code for ground truth images
label_colors = [(64,128,64),(192,0,128),(0,128,192),(0,128,64),(128,0,0),(64,0,128),(64,0,192),(192,128,64),(192,192,128),(64,64,128),(128,0,192),(192,0,64),(128,128,64),(192,0,192),(128,64,64),(64,192,128),(64,64,0),(128,64,128),(128,128,192),(0,0,192),(192,128,128),(128,128,128),(64,128,192),(0,0,64),(0,64,64),(192,64,128),(128,128,0),(192,128,192),(64,0,64),(192,192,0),(0,0,0),(64,192,0)]
label_class = [255,255,0,255,1,2,255,255,3,4,255,255,255,255,255,255,5,6,255,7,8,9,255,255,255,255,10,255,255,255,255,255]

f = open('camvid_gt_train.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camvid_train-gt-augmented-lmdb', map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.NEAREST) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        # convert to one dimensional ground truth labels
        tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
        for i in range(0,len(label_colors)):
            tmp[:,:,0] = tmp[:,:,0] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)

        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

#Cropping
f = open('camvid_gt_train.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camvid_train-gt-augmented-lmdb', map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        im = Image.open(in_) # load image
        im = im.crop((left,top,right,bottom))
        im = np.array(im) # convert to nparray you need
        # convert to one dimensional ground truth labels
        tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
        for i in range(0,len(label_colors)):
            tmp[:,:,0] = tmp[:,:,0] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)

        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

#Mirroring
f = open('camvid_gt_train.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camvid_train-gt-augmented-lmdb', map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        im = Image.open(in_) # load image
        im = ImageOps.mirror(im)
        im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.NEAREST) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        # convert to one dimensional ground truth labels
        tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
        for i in range(0,len(label_colors)):
            tmp[:,:,0] = tmp[:,:,0] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)

        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

#Cropping & Mirroring
f = open('camvid_gt_train.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camvid_train-gt-augmented-lmdb', map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        im = Image.open(in_) # load image
        im = im.crop((left,top,right,bottom))
        im = ImageOps.mirror(im)
        im = np.array(im) # convert to nparray you need
        # convert to one dimensional ground truth labels
        tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
        for i in range(0,len(label_colors)):
            tmp[:,:,0] = tmp[:,:,0] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)

        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camvid train gt done'

ridx = np.arange(233)
np.random.shuffle(ridx)

f = open('camvid_test.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camvid_test-lmdb', map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.ANTIALIAS) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camvid test done'

f = open('camvid_gt_test.txt','r')
inputs = f.read().splitlines()
inputs = [inputs[i] for i in ridx]
f.close()

in_db = lmdb.open('camvid_test-gt-lmdb', map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        im = Image.open(in_) # load image
        im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.NEAREST) # downsize for reduced memory usage
        im = np.array(im) # convert to nparray you need
        # convert to one dimensional ground truth labels
        tmp = np.uint8(np.zeros((im.shape[0],im.shape[1],1)))
        for i in range(0,len(label_colors)):
            tmp[:,:,0] = tmp[:,:,0] + label_class[i]*np.prod(np.equal(im,label_colors[i]),2)

        # - in Channel x Height x Width order (switch from H x W x C)
        tmp = tmp.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(tmp)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()

print 'camvid test gt done'