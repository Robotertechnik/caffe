from PIL import Image

i = 0
while i < 243:
    im = Image.open('inputs/argmax-'+str(i+1).zfill(5)+'.pgm')
    im = im.resize((int(im.size[0]*0.25),int(im.size[1]*0.25)),Image.NEAREST)
    im.save('small_inputs/argmax-'+str(i+1).zfill(5)+'.pgm')
    i = i +1
