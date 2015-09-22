from PIL import Image
from PIL import ImageOps

w, h = 960, 720
left = (w - 480)/2
top = (h - 360)/2
right = (w + 480)/2
bottom = (h + 360)/2
#Resize
f = open('../../data/camvid/LabeledApproved_full/camvid_gt_train.txt','r')
inputs = f.read().splitlines()
f.close()
for in_idx, in_ in enumerate (inputs):
	im = Image.open('../../data/camvid/LabeledApproved_full/'+in_)
	im = ImageOps.mirror(im)
	im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.NEAREST)
	im.save('../../data/camvid/LabeledApproved_full/'+'R'+in_)
#Cropping + Mirroring
f = open('../../data/camvid/LabeledApproved_full/camvid_gt_train.txt','r')
inputs = f.read().splitlines()
f.close()
for in_idx, in_ in enumerate (inputs):
	im = Image.open('../../data/camvid/LabeledApproved_full/'+in_)
	im = ImageOps.mirror(im)
	im = im.crop((left,top,right,bottom))
	im.save('../../data/camvid/LabeledApproved_full/'+'CM'+in_)
#Mirroring
f = open('../../data/camvid/LabeledApproved_full/camvid_gt_train.txt','r')
inputs = f.read().splitlines()
f.close()
for in_idx, in_ in enumerate (inputs):
	im = Image.open('../../data/camvid/LabeledApproved_full/'+in_)
	im = ImageOps.mirror(im)
	im = im.resize((int(im.size[0]*0.5),int(im.size[1]*0.5)),Image.NEAREST)
	im.save('../../data/camvid/LabeledApproved_full/'+'M'+in_)
#Cropping
f = open('../../data/camvid/LabeledApproved_full/camvid_gt_train.txt','r')
inputs = f.read().splitlines()
f.close()
for in_idx, in_ in enumerate (inputs):
	im = Image.open('../../data/camvid/LabeledApproved_full/'+in_)
	im = im.crop((left,top,right,bottom))
	im.save('../../data/camvid/LabeledApproved_full/'+'C'+in_)