import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif

from schema.mpanze_widefield import ReferenceImage

image_0 = tif.imread("F:/Jithin/Batch_6/M42/Widefield/Pre_Stroke/Widefield_session1_forelimb.tif", key=0)

plt.figure()
plt.imshow(image_0, cmap="Greys_r")
plt.figure()
image_crop = image_0[65:65+385, 100:100+385]
plt.imshow(image_crop, cmap="Greys_r")

image_ref = cv2.resize(image_crop, (512,512))
plt.figure()
plt.imshow(image_ref,cmap="Greys_r")

tif.imwrite("F:/Jithin/Batch_6/M42/Widefield/Pre_Stroke/img_ref.tif", image_ref)
mask = tif.imread("F:/Jithin/Batch_6/M42/Widefield/Pre_Stroke/Mask.tif")

# check mask
# N.B.!!!!! when mask is True (or 255), data is masked
mask = 255-mask

plt.figure()
plt.imshow(np.ma.masked_array(image_ref, mask= mask))

ReferenceImage().insert1({"username": "jnambi", "mouse_id": 42, "led_colour": "Blue", "ref_date": "2021-07-04", "ref_image": image_ref, "ref_mask": mask})