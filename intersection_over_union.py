
#w, h are reference bounding box size
#return variables are matrices, you can get IoU by calling SI/SU
def compute_pascal_mapping(pd,w,h,imwidth,imheight):

    boxes = np.mgrid[0:imheight,0:imwidth]

    bx1 = boxes[1]-w/2

    by1 = boxes[0]-h/2

    bx2 = boxes[1]+w/2

    by2 = boxes[0]+h/2

    SA=h*w

    SB = (pd[3]-pd[1])*(pd[2]-pd[0])

    min2 = np.minimum(bx2,pd[2])

    max0 = np.maximum(bx1,pd[0])

    min3 = np.minimum(by2,pd[3])

    max1 = np.maximum(by1,pd[1])

    SI=np.maximum(0,min2-max0)*np.maximum(0,min3-max1)

    SI=SI.astype(np.float)

    SU=SA+SB-SI

    return SI,SU,SA,SB