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

#net = caffe.Net('../googlenet_12outputs/late_deploy.prototxt', '../googlenet_12outputs/googlenet_8_12.caffemodel', caffe.TEST)
net = caffe.Net('deploy.prototxt', 'GoogLeNet_8s_PT_DA_LB_iter_25000.caffemodel', caffe.TEST)
for in_idx, in_ in enumerate(inputs):
    im = Image.open('../../data/camvid/701_StillsRaw_full/'+in_+'.png')
    #im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.ANTIALIAS) #The program crashes
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
    scipy.misc.imsave('inputs/argmax-'+str(in_idx+1).zfill(5)+'.pgm' , out)
    im = Image.open('inputs/argmax-'+str(in_idx+1).zfill(5)+'.pgm')
    pix = im.load() #converts image
    y = 0
    pix2 = np.zeros((im.size[0],im.size[1]))
    print 'converting argmax'
    while y < im.size[1] :
        x = 0
        while x < im.size[0]:
            if pix[x,y] == 153:
                pix2[x,y] = 8 #Road
            elif pix[x,y] == 25:
                pix2[x,y] = 2 #Building
            elif pix[x,y] == 229:
                pix2[x,y] = 11 #Sky
            elif pix[x,y] == 255:
                pix2[x,y] = 6 #Tree
            elif pix[x,y] == 178:
                pix2[x,y] = 9 #Sidewalk
            elif pix[x,y] == 51:
                pix2[x,y] = 3 #Car
            elif pix[x,y] == 76:
                pix2[x,y] = 4 #Column Pole
            elif pix[x,y] == 204:
                pix2[x,y] = 10 #Sign Symbol
            elif pix[x,y] == 102:
                pix2[x,y] = 5 #Fence
            elif pix[x,y] == 127:
                pix2[x,y] = 7 #Pedestrian
            elif pix[x,y] == 0:
                pix2[x,y] = 1 #Byciclist
            else:
                pix2[x,y] = 12 #Void
            x = x + 1
        y = y +1
        print 'image done'
    scipy.misc.imsave('inputs/camvid_argmax-'+str(in_idx+1).zfill(5)+'.pgm' , pix)
print 'camvid done'



