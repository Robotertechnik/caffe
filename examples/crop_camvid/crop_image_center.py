#!/usr/bin/env python
import numpy
from PIL import Image

# im = Image.open(args.input_image_directory+file)
im = Image.open('0016E5_07997.png')
w, h = im.size
left = (w - 480)/2
top = (h - 360)/2
right = (w + 480)/2
bottom = (h + 360)/2

out = im.crop((left, top, right, bottom))

out.save('cropedtest.png')