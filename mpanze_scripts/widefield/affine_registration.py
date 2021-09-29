"""
Script for performing spatial registration using (partial) affine transforms.
Script finds RawImagingFile entries which have not yet been found and aligns them to the reference Image
"""

import cv2

import login
from schema.mpanze_widefield import ReferenceImage, RawImagingFile, AffineRegistration
import matplotlib.pyplot as plt
import numpy as np

## file selection: use 1st Reference image, and any Unaligned reference file

# load ref image
mouse = {"username": "jnambi", "mouse_id": 42}
ref_pk = (ReferenceImage() & mouse).fetch("KEY", limit=1, as_dict=True)[0]
ref_img = (ReferenceImage() & ref_pk).fetch1("ref_image")

# get first RawImagingFile not in AffineParameters, and load first image
login.set_working_directory("F:/Jithin/")
file_pk = (RawImagingFile() - AffineRegistration() & mouse).fetch("KEY", as_dict=True, limit=1)[0]
reg_img = (RawImagingFile & file_pk).get_first_image()

plt.subplot(1,2,1)
plt.imshow(ref_img, "Greys_r")
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(reg_img, "Greys_r")
plt.colorbar()
plt.tight_layout()

## choose at least 2 points per image to register
posList=[]
def onMouse(event, x, y, flags, param):
   global posList
   if event == cv2.EVENT_LBUTTONDOWN:
       print(x,y)
       posList.append((x, y))

#auto adjust luminosity for visual clarity
img_ref = np.copy(ref_img)
reg_img = np.copy(reg_img)
img_ref = (img_ref-np.amin(img_ref))/(np.amax(img_ref)-np.amin(img_ref))
reg_img = (reg_img-np.amin(reg_img))/(np.amax(reg_img)-np.amin(reg_img))

cv2.namedWindow("Ref")
cv2.setMouseCallback('Ref', onMouse)
cv2.namedWindow("Reg")
cv2.setMouseCallback('Reg', onMouse)

while(1):
    cv2.imshow('Ref', img_ref)
    cv2.imshow('Reg', reg_img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

# apply affine transformation to images, if the overlay doesn't look right, use different set of coordinates
pts_dest = np.float32(posList[::2])
pts_src = np.float32(posList[1::2])

M,_ = cv2.estimateAffinePartial2D(pts_src,pts_dest)
dst = cv2.warpAffine(reg_img, M, (512, 512))
plt.figure()
plt.imshow(img_ref, cmap="Reds_r", alpha=0.5)
plt.imshow(dst, "Blues_r", alpha=0.5)

## save params
new_key = {**ref_pk, **file_pk, "affine_matrix": M}
AffineRegistration().insert1(new_key)