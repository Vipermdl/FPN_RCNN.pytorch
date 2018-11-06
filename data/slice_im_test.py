# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import os
import os.path as osp
from PIL import Image
from xml.dom.minidom import Document
import xml.etree.ElementTree as ET

def slice_im(img_path, source_img_path, dest_img_path, sliceHeight=1056, sliceWidth=1408, zero_frac_thresh=0.2,overlap=0.2, verbose=False, version = True):

    image_name = img_path.replace(source_img_path+"/", "").replace(".jpg", "")

    os.mkdir(osp.join(dest_img_path, image_name))

    dest_img_path = osp.join(dest_img_path, image_name)

    image0 = cv2.imread(os.path.join(source_img_path, image_name + '.jpg'), 1)
    win_h, win_w = image0.shape[:2]

    pad = 0
    if sliceHeight > win_h:
        pad = sliceHeight - win_h
    if sliceWidth > win_w:
        pad = max(pad, sliceWidth - win_w)
    # pad the edge of the image with black pixels
    if pad > 0:    
        border_color = (0, 0, 0)
        image0 = cv2.copyMakeBorder(image0, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border_color)

    win_size = sliceHeight * sliceWidth

    n_ims = 0
    n_ims_nonull = 0
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    for y0 in range(0, image0.shape[0], dy):#sliceHeight):
        for x0 in range(0, image0.shape[1], dx):#sliceWidth):
            n_ims += 1
            
            # make sure we don't have a tiny image on the edge
            if y0 + sliceHeight > image0.shape[0]:
                y = image0.shape[0] - sliceHeight
            else:
                y = y0
            if x0 + sliceWidth > image0.shape[1]:
                x = image0.shape[1] - sliceWidth
            else:
                x = x0

            # extract image
            window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
            # get black and white image
            window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)
            # find threshold that's not black
            # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=threshold
            ret,thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
            non_zero_counts = cv2.countNonZero(thresh1)
            zero_counts = win_size - non_zero_counts
            zero_frac = float(zero_counts) / win_size
            if zero_frac >= zero_frac_thresh:
                if verbose:
                    print("Zero frac too high at:", zero_frac)
                continue
            else:
                out_file_name = str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                out_img_path = osp.join(dest_img_path, out_file_name+".jpg")

                cv2.imwrite(out_img_path, window_c)
                n_ims_nonull += 1
    return


if __name__ == '__main__':

    source_img_path = './guangdong_round2_test_b_20181106'
    dest_img_path = './test_patch'

    if osp.exists(dest_img_path):
        pass
    else:
        os.mkdir(dest_img_path)

    img_list = glob.glob(osp.join(source_img_path, "*.jpg"))

    for img_path in img_list:
        slice_im(img_path, source_img_path, dest_img_path)
    print("Done")


