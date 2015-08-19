#!/usr/bin/env sh
ffmpeg -i Google\ Drive/A\&R\ TU\ Dortmund/4th\ Semester\ 2015/Master\ thesis/Computer\ Vision\ Thesis/CamVid\ Database/Videos/0005VD.MXF -s 720x960 image%05d.png
./caffe-master/segmentation/print_segmentation.py