import cv2
import os
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import scipy.signal
import scipy.ndimage
import vifp


def cal_ssim(grayA, grayB):
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    return score

def valuelist(size):
    object = list()
    for i in range(size):
        object.append( list() ) 
    return object

file_style = os.listdir('/home/jwsi/PIX2PIX/compressedData/recon_95/back/')
file_style.sort()

file_content = os.listdir('/home/jwsi/PIX2PIX/compressedData/recon_orig/')
file_content.sort()


count = 0

psnr_content = [0 for i in range(3)]
ssim_content = [0 for i in range(3)]
vifp_content = [0 for i in range(3)]


for i in range(40):
    img1 = cv2.imread('/home/jwsi/PIX2PIX/compressedData/recon_orig/'+file_content[i])
    img3 = cv2.imread('/home/jwsi/PIX2PIX/output_test/back95/'+ file_style[i])
    img4 = cv2.imread('/home/jwsi/PIX2PIX/output_test/back98/'+file_style[i])

    psnr_content[0] += cv2.PSNR(img1, img1)
    psnr_content[1] += cv2.PSNR(img1, img3)
    psnr_content[2] += cv2.PSNR(img1, img4)
 
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
  
    ssim_content[0] += cal_ssim(img1, img1)
    ssim_content[1] += cal_ssim(img1, img3)
    ssim_content[2] += cal_ssim(img1, img4)
 
    ref = scipy.misc.imread('/home/jwsi/PIX2PIX/compressedData/recon_orig/'+file_content[i], flatten=True).astype(np.float32)
    dist2 = scipy.misc.imread('/home/jwsi/PIX2PIX/output_test/back95_pix2pix/'+ file_style[i], flatten=True).astype(np.float32)
    dist3 = scipy.misc.imread('/home/jwsi/PIX2PIX/output_test/back98_pix2pix/'+ file_style[i], flatten=True).astype(np.float32)

    vifp_content[0]+=vifp.vifp_mscale(ref, ref)
    vifp_content[1]+=vifp.vifp_mscale(ref, dist2)
    vifp_content[2]+=vifp.vifp_mscale(ref, dist3)
  
    count += 1



contentloss = [x/count for x in psnr_content]
contentloss2 = [x/count for x in ssim_content]
vifploss = [x/count for x in vifp_content]


for i in range(3):  
    print("{:>7.3f}".format(contentloss[i]), end= "        ")
print()

for i in range(3):  
    print("{:>7.3f}".format(contentloss2[i]), end= "        ")
print()

for i in range(3):  
    print("{:>7.3f}".format(vifploss[i]), end= "        ")
print()

