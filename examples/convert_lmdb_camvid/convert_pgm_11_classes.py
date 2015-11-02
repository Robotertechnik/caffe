from PIL import Image
import numpy as np
import scipy
import cv2

f = open('../camvid_groundtruth.txt','r')
inputs = f.read().splitlines()
f.close()

#for in_idx, in_ in 599:
for in_idx, in_ in enumerate(inputs):
	#im = Image.open('/Users/CarlosTrevino/caffe-master/examples/convert_lmdb_camvid/inputs/argmax-'+str(in_idx+1).zfill(5))+'.pgm')
    im = Image.open('inputs/argmax-'+str(in_idx+1).zfill(5)+'.pgm')
    pix = im.load() #converts image
    y = 0
    pix2 = np.zeros((im.size[0],im.size[1]))
    while y < im.size[1] :
    #while y < 720 :
        x = 0
        while x < im.size[0]:
        #while x < 960:
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
        #print y
    #pix3 = scipy.misc.toimage(pix2)
    #A = Image.fromarray(pix2)
    pix2 = np.transpose(pix2)
    #print A.size
    #scipy.misc.imsave('inputs/camvid_argmax-'+str(in_idx+1).zfill(5)+'.pgm',pix2)
    # A = Image.fromarray(pix2)
    #im.show(A)
    #A = A.convert('RGB')
    cv2.imwrite('inputs/camvid_argmax-'+str(in_idx+1).zfill(5)+'.pgm',pix2)
    print 'image '+str(in_idx+1).zfill(5)+'done'
    #A.save('inputs/camvid_argmax-'+str(in_idx+1).zfill(5)+'.pgm')

