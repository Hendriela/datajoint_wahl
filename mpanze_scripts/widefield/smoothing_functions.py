import cv2
import numpy as np
import json


def gaussian_blur_2d(stack, params):
    """
    Smoothes image stack with gaussian kernel. works frame by frame only on x and y.
    Args:
        stack: image stack to be smoothed of size (frames, x, y)
        params: params to pass to function in .json string format
                size_x, size_y are the size of the kernel. must be odd

    Returns: blurred stack

    """
    # load params
    p = json.loads(params)
    size_x, size_y = p["size_x"], p["size_y"]
    # initialize stack
    stack_blur = np.empty_like(stack)

    for i, frame in enumerate(stack):
        stack_blur[i] = cv2.GaussianBlur(frame, (size_x, size_y), 0)
    return stack_blur
