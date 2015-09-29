import numpy as np
from PIL import Image

f = open('camvid_train.txt','r')
inputs = f.read().splitlines()
f.close()

for in_idx, in_ in enumerate(inputs):
    im = Image.open(in_)
    im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.ANTIALIAS)
    im.save('small_inputs/input'+str(in_idx+1).zfill(5)+'.ppm')

f = open('camvid_test.txt','r')
inputs = f.read().splitlines()
f.close()

for in_idx, in_ in enumerate(inputs):
    im = Image.open(in_)
    im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.ANTIALIAS)
    im.save('small_inputs/input'+str(in_idx+368).zfill(5)+'.ppm')