from PIL import Image
from PIL import ImageOps

w, h = 960, 720
left = (w - 480)/2
top = (h - 360)/2
right = (w + 480)/2
bottom = (h + 360)/2

#Cropping Center
f = open('../../data/camvid/701_StillsRaw_full/camvid_train.txt','r')
inputs = f.read().splitlines()
f.close()
for in_idx, in_ in enumerate (inputs):
	im = Image.open('../../data/camvid/701_StillsRaw_full/'+in_)
	im = im.crop((left,top,right,bottom))
	im.save('../../data/camvid/701_StillsRaw_full/'+'C'+in_)

left = 480
top = 720
right = 960
bottom = 360
#Cropping Upper Right
f = open('../../data/camvid/701_StillsRaw_full/camvid_train.txt','r')
inputs = f.read().splitlines()
f.close()
for in_idx, in_ in enumerate (inputs):
	im = Image.open('../../data/camvid/701_StillsRaw_full/'+in_)
	im = im.crop((left,bottom,right,top))
	im.save('../../data/camvid/701_StillsRaw_full/'+'UR'+in_)

left = 0
top = 720
right = 480
bottom = 360
#Cropping Upper Left
f = open('../../data/camvid/701_StillsRaw_full/camvid_train.txt','r')
inputs = f.read().splitlines()
f.close()
for in_idx, in_ in enumerate (inputs):
	im = Image.open('../../data/camvid/701_StillsRaw_full/'+in_)
	im = im.crop((left,bottom,right,top))
	im.save('../../data/camvid/701_StillsRaw_full/'+'UL'+in_)

left = 480
top = 360
right = 960
bottom = 0
#Cropping Upper Right
f = open('../../data/camvid/701_StillsRaw_full/camvid_train.txt','r')
inputs = f.read().splitlines()
f.close()
for in_idx, in_ in enumerate (inputs):
	im = Image.open('../../data/camvid/701_StillsRaw_full/'+in_)
	im = im.crop((left,bottom,right,top))
	im.save('../../data/camvid/701_StillsRaw_full/'+'LR'+in_)

left = 0
top = 360
right = 480
bottom = 0
#Cropping Upper Left
f = open('../../data/camvid/701_StillsRaw_full/camvid_train.txt','r')
inputs = f.read().splitlines()
f.close()
for in_idx, in_ in enumerate (inputs):
	im = Image.open('../../data/camvid/701_StillsRaw_full/'+in_)
	im = im.crop((left,bottom,right,top))
	im.save('../../data/camvid/701_StillsRaw_full/'+'LL'+in_)