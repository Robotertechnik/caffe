from PIL import Image
import numpy as np

#Load txt file with file names
f = open('camvid_groundtruth.txt','r')
inputs = f.read().splitlines()
f.close()

#
for in_idx, in_ in enumerate (inputs):
    im = Image.open('../data/camvid/LabeledApproved_full/'+in_+'_L.png') # load image
    im = im.resize((int(im.size[0]*0.25),int(im.size[1]*0.25)),Image.NEAREST) #Downsampling
    txt = open('convert_lmdb_camvid/small_inputs/input-'+str(in_idx+1).zfill(5)+'.txt','w') #creates txt file for the groundtruth
    y = 0 # initializes y counter
    pix = im.load() #converts image
    while y < im.size[1] :
        x = 0
        while x < im.size[0]:
            if pix[x,y] == (128,64,128):
                txt.write('11') #Road
            elif pix[x,y] == (128, 0, 0):
                txt.write('1') #Building
            elif pix[x,y] == (128, 128, 128):
                txt.write('2') #Sky
            elif pix[x,y] == (128, 128, 0):
                txt.write('3') #Tree
            elif pix[x,y] == (0, 0, 192):
                txt.write('4') #Sidewalk
            elif pix[x,y] == (64, 0, 128):
                txt.write('5') #Car
            elif pix[x,y] == (192, 192, 128):
                txt.write('6') #Column Pole
            elif pix[x,y] == (192, 128, 128):
                txt.write('7') #Sign Symbol
            elif pix[x,y] == (64, 64, 128):
                txt.write('8') #Fence
            elif pix[x,y] == (64, 64, 0):
                txt.write('9') #Pedestrian
            elif pix[x,y] == (0, 128, 192):
                txt.write('10') #Biciclyst
            else:
                txt.write('-1') #Void
            x = x+1
            if x < im.size[0]:
                txt.write(' ')
            else:
                txt.write('\n')
        y = y + 1
    txt.close()








