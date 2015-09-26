#!/bin/bash

CAFFE_ROOT=~/caffe-master

python $CAFFE_ROOT/tools/extra/plot_log.py 6 models/bvlc_googlenet/FCN_GoogLeNet_32s/training_loss.png models/bvlc_googlenet/FCN_GoogLeNet_32s/caffe.log