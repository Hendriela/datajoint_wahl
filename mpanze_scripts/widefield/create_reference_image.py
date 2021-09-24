import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif

from schema.mpanze_widefield import ReferenceImage

image_0 = tif.imread("F:/Jithin/Batch_6/M41/Widefield/Pre_Stroke/Hindlimb_1.tif", key=0)

plt.figure()
plt.imshow(image_0, cmap="Greys_r")
plt.figure()
image_crop = image_0[50:50+435, 50:50+435]
plt.imshow(image_crop, cmap="Greys_r")

image_ref = cv2.resize(image_crop, (512,512))
plt.figure()
plt.imshow(image_ref,cmap="Greys_r")

tif.imwrite("F:/Jithin/Batch_6/M41/Widefield/Pre_Stroke/img_ref.tif", image_ref)
mask = tif.imread("F:/Jithin/Batch_6/M41/Widefield/Pre_Stroke/img_ref.tif")

ReferenceImage().insert1({"username": "jnambi", "mouse_id": 41, "led_colour": "Blue", "ref_date": "2021-07-04", "ref_image": image_ref, "ref_mask": mask})