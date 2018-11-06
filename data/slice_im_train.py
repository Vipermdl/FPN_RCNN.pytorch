# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import os
import os.path as osp
from PIL import Image
from xml.dom.minidom import Document
import xml.etree.ElementTree as ET

def cal_IOU(box1, box2):
    """
    box1, box2: list or numpy array of size 4*2 or 8, h_index first
    """
    box1 = [box1[0], box1[1], box1[2], box1[1], box1[2], box1[3], box1[0], box1[3]]
    box2 = [box2[0], box2[1], box2[2], box2[1], box2[2], box2[3], box2[0], box2[3]]

    box1 = np.array(box1, dtype=np.int).reshape([1, 4, 2])
    box2 = np.array(box2, dtype=np.int).reshape([1, 4, 2])
    box1_max = box1.max(axis=1)
    box2_max = box2.max(axis=1)
    w_max = int(max(box1_max[0][0], box2_max[0][0]))
    h_max = int(max(box1_max[0][1], box2_max[0][1]))
    canvas = np.zeros((h_max + 1, w_max + 1))
    # print(canvas.shape)
    box1_canvas = canvas.copy()
    box1_area = np.sum(cv2.drawContours(box1_canvas, box1, -1, 1, thickness=-1))
    # print(box1_area)
    box2_canvas = canvas.copy()
    box2_area = np.sum(cv2.drawContours(box2_canvas, box2, -1, 1, thickness=-1))
    # print(box2_area)
    cv2.drawContours(canvas, box1, -1, 1, thickness=-1)
    cv2.drawContours(canvas, box2, -1, 1, thickness=-1)
    union = np.sum(canvas)

    intersction = box1_area + box2_area - union
    return intersction / union

def refine_bbox(bbox1, bbox2):
    bbox = list()
    bbox.append(max(bbox1[0], bbox2[0]))
    bbox.append(max(bbox1[1], bbox2[1]))
    bbox.append(min(bbox1[2], bbox2[2]))
    bbox.append(min(bbox1[3], bbox2[3]))
    return bbox

def _load_pascal_annotation(filename):
    tree = ET.parse(filename)
    objs = tree.findall('object')
    objects = list()
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        class_name = obj.find('name').text.lower().strip()
        objects.append([x1, y1, x2, y2, class_name])
    return objects

def check_point(bbox, width, height, image_name): 
    if bbox[0] <= 0:
        bbox[0] = 1
    if bbox[1] <= 0:
        bbox[1] = 1
    if bbox[2] <= 0:
        print(image_name, bbox)
        exit()
    if bbox[3] <= 0:
        print(image_name, bbox)
        exit()
    return bbox

def writeInfoToXml(objects, information, out_xml_path):
    doc = Document()
    annotion = doc.createElement("annotion")
    doc.appendChild(annotion)
    folder = doc.createElement("folder")
    folder_text = doc.createTextNode("LVCai")
    folder.appendChild(folder_text)
    annotion.appendChild(folder)
    filename = doc.createElement("filename")
    filename_text = doc.createTextNode(str(information[0]))
    filename.appendChild(filename_text)
    annotion.appendChild(filename)
    source = doc.createElement("source")
    database = doc.createElement("database")
    database_text = doc.createTextNode("The LVCai Dataset")
    database.appendChild(database_text)
    source.appendChild(database)
    annotation = doc.createElement("annotation")
    annotation_text = doc.createTextNode("The TianChi competition of Aluminum-surface-surface-identification")
    annotation.appendChild(annotation_text)
    source.appendChild(annotation)
    image = doc.createElement("image")
    image_text = doc.createTextNode("flickr")
    image.appendChild(image_text)
    source.appendChild(image)
    annotion.appendChild(source)
    size = doc.createElement("size")
    width = doc.createElement("width")
    width_text = doc.createTextNode(str(information[1]))
    width.appendChild(width_text)
    size.appendChild(width)
    height = doc.createElement("height")
    height_text = doc.createTextNode(str(information[2]))
    height.appendChild(height_text)
    size.appendChild(height)
    depth = doc.createElement("depth")
    depth_text = doc.createTextNode("3")
    depth.appendChild(depth_text)
    size.appendChild(depth)
    annotion.appendChild(size)
    segmented = doc.createElement("segmented")
    segmented_text = doc.createTextNode("0")
    segmented.appendChild(segmented_text)
    annotion.appendChild(segmented)

    for i in range(len(objects)):
        object = doc.createElement("object")

        name = doc.createElement("name")
        name_text = doc.createTextNode(str(objects[i][-1]))
        name.appendChild(name_text)
        object.appendChild(name)

        pose = doc.createElement("pose")
        pose_text = doc.createTextNode("Unspecified")
        pose.appendChild(pose_text)
        object.appendChild(pose)

        truncated = doc.createElement("truncated")
        truncated_text = doc.createTextNode("1")
        truncated.appendChild(truncated_text)
        object.appendChild(truncated)

        difficult = doc.createElement("difficult")
        difficult_text = doc.createTextNode(str(0))
        difficult.appendChild(difficult_text)
        object.appendChild(difficult)

        bndbox = doc.createElement("bndbox")

        xmin = doc.createElement("xmin")
        xmin_text = doc.createTextNode(str(objects[i][0]))
        xmin.appendChild(xmin_text)
        bndbox.appendChild(xmin)

        ymin = doc.createElement("ymin")
        ymin_text = doc.createTextNode(str(objects[i][1]))
        ymin.appendChild(ymin_text)
        bndbox.appendChild(ymin)

        xmax = doc.createElement("xmax")
        xmax_text = doc.createTextNode(str(objects[i][2]))
        xmax.appendChild(xmax_text)
        bndbox.appendChild(xmax)

        ymax = doc.createElement("ymax")
        ymax_text = doc.createTextNode(str(objects[i][3]))
        ymax.appendChild(ymax_text)
        bndbox.appendChild(ymax)

        object.appendChild(bndbox)
        annotion.appendChild(object)

    with open(out_xml_path, 'wb+') as f:
        f.write(doc.toprettyxml(indent="\t", newl = "\n", encoding="utf-8"))
        f.close()
    return

def visualize_label(img, boxes, color=(0, 255, 0)):
    """
    img: HWC
    boxes: array of num * 4 * 2
    """
    boxes = np.array(boxes)[:, :-1]
    # boxes = [int(term) for term in box for box in boxes]

    temp_boxes = list()
    for box in boxes:
        temp_box = list()
        for term in box:
            temp_box.append(int(term))
        temp_boxes.append(temp_box)

    boxes = np.array(temp_boxes).reshape(-1, 4, 2)
    img = np.ascontiguousarray(img)
    cv2.drawContours(img, boxes, -1, color, thickness=1)
    return img

def slice_im(source_xml, source_img_path, source_xml_path, dest_img_path,  dest_xml_path, writer, sliceHeight=1056, sliceWidth=1408, zero_frac_thresh=0.2, overlap=0.1, verbose=False, version = True):
    """
    :param source_xml:
    :param source_img_path:
    :param source_xml_path:
    :param dest_img_path:
    :param dest_xml_path:
    :param sliceHeight:
    :param sliceWidth:
    :param zero_frac_thresh:
    :param overlap:
    :param verbose:
    :param version:
    :return: slice the image of 800 * 800, the number of bbox may be increased
    """
    image_name = source_xml.replace(source_xml_path+"/", "").replace(".xml", "")

    objects = _load_pascal_annotation(source_xml)

    if len(objects) is None:
        return

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

            # make new labels
            new_labels = list()

            patch_bbox = [x, y, x + sliceWidth, y + sliceHeight]
            for objects_ in objects:
                if cal_IOU(patch_bbox, objects_[0:4]) != 0:
                    bbox = refine_bbox(patch_bbox, objects_[0:4])
                    
                    if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                        pass
                    else:
                        continue
                    
                    bbox[0] = bbox[0] - x
                    bbox[1] = bbox[1] - y
                    bbox[2] = bbox[2] - x
                    bbox[3] = bbox[3] - y
                    
                    bbox = check_point(bbox, sliceWidth, sliceHeight, image_name)
                    
                    bbox.append(objects_[-1])
                    new_labels.append(bbox)

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
                out_file_name = image_name + '_' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                out_img_path = osp.join(dest_img_path, out_file_name+".jpg")
                out_xml_path = osp.join(dest_xml_path, out_file_name+".xml")

                if verbose:
                    print("outpath:", out_file_name)

                if len(new_labels) == 0:
                    continue
                else:
                    # window_c = visualize_label(window_c, new_labels, color=(255, 0, 0))
                    if version:
                        information = [out_file_name, sliceWidth, sliceHeight]
                        writeInfoToXml(new_labels, information, out_xml_path)
                        
                        writer.write(out_file_name+"\n")
                        
                        cv2.imwrite(out_img_path, window_c)
                n_ims_nonull += 1
    return


if __name__ == '__main__':

    source_img_path = './VOCdevkit/VOC2012/JPEGImages'
    source_xml_path = './VOCdevkit/VOC2012/Annotations'
    dest_xml_path = './VOCdevkit_patch/VOC2012/Annotations'
    dest_img_path = './VOCdevkit_patch/VOC2012/JPEGImages'

    main_path = './VOCdevkit_patch/VOC2012/ImageSets/Main'

    xml_list = glob.glob(osp.join(source_xml_path, "*.xml"))
    with open(osp.join(main_path, 'trainval.txt'), 'a+') as writer:
      for xml_ in xml_list:
          slice_im(xml_, source_img_path=source_img_path,source_xml_path=source_xml_path, dest_img_path=dest_img_path, dest_xml_path=dest_xml_path, writer=writer)
    writer.close()
    print("Done")


