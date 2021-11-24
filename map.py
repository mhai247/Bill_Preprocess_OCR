import cv2
import numpy as np
from pyimagesearch import transform
import glob
from config import out_rot_img_dir, out_det_txt_dir, out_map_img_dir, out_map_txt_dir
import logging
import time
import os

def overlap(box1, box2):
    # print(box1, box2)
    if max(box1[:, 1]) <= min(box2[:, 1]):
        return 0
    if min(box1[:, 1]) >= max(box2[:, 1]):
        return 0
    if max(box1[:, 0]) <= min(box2[:, 0]):
        return 0
    if min(box1[:, 0]) >= max(box2[:, 0]):
        return 0
    return 1

files = glob.glob(out_rot_img_dir + '/*.jpg')
for file in files:
    img_file_name = file.split('/')[-1]
    txt_file_name = img_file_name.replace('.jpg', '.txt')
    
    img = cv2.imread(file)
    txt_file_path = os.path.join(out_det_txt_dir, txt_file_name)
    out_img_file_path = os.path.join(out_map_img_dir, img_file_name)
    out_txt_file_path = os.path.join(out_map_txt_dir, txt_file_name)

    txt_file = open(txt_file_path, 'r')

    start = time.time()

    height, width = img.shape[:2]
    blank_image = np.ones((height,width), np.uint8) * 255

    MAX_DIS = height/70

    new_box = np.zeros((4,2), dtype=np.int16)
    boxes = np.zeros((100,4,2), dtype=np.int16)
    # plus = 0

    line = []
    lines = txt_file.readlines()
    txt_file.close()

    num_lines = len(lines)
    for i in range(num_lines):
        line = lines[num_lines-i-1]
        idxs = line.split(',')
        pts = []
        for k in range(8):
            pts.append(int(idxs[k]))

        pts = np.reshape(pts, (4,2))
        warp = transform.four_point_transform(img, pts)

        gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

        # sharpen image
        sharpen = cv2.GaussianBlur(gray, (0,0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        # # apply adaptive threshold to get black and white effect
        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 30)

        w,h = warp.shape[:2]
        tl = pts[0]

        new_box = np.array( [[tl[1], tl[0]], [tl[1],tl[0] + h], [tl[1] + w, tl[0]+h], [tl[1] + w, tl[0]]])

        #align
        if i > 0 and abs(boxes[i-1][0][0] - tl[1]) < MAX_DIS:
            do = 1
            add = boxes[i-1][0][0] - tl[1]
            change_box = np.array( [[add + tl[1], tl[0]], [add + tl[1],tl[0] + h], [add + tl[1] + w, tl[0]+h], [add + tl[1] + w, tl[0]]])
            for j in range(min(i-1, 5)):
                if overlap(boxes[i-j-1], change_box):
                    do = 0
                    break
            if do:
                tl[1] = boxes[i-1][0][0]

        #prevent overlaping
        max_h = tl[1]
        for j in range(min(i-1, 5)):
            if overlap(boxes[i-j-1], new_box):
                if boxes[i-j-1][3][0] > max_h:
                    max_h = boxes[i-j-1][3][0]

        tl[1] = max_h


        if tl[0] + h >= width:
            # out = tl[0] + h - width + 1
            tl[0] = width - h - 1
        # print(blank_image.shape)

        if tl[1] + w >= height:
            continue
        dilate = 0

        blank_image[tl[1]+dilate:tl[1]+w-dilate, tl[0]+dilate:tl[0]+h- dilate] = thresh[dilate:w-dilate, dilate:h-dilate]

        boxes[i] = np.array([[tl[1], tl[0]], [tl[1],tl[0] + h], [tl[1] + w, tl[0]+h], [tl[1] + w, tl[0]]])

        # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        # cv2.imshow('output', blank_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    end = time.time()

    cv2.imwrite(out_img_file_path, blank_image)
    logging.info("Saved image in {}".format(out_img_file_path))
    logging.info("Time: {}\n".format(end-start))
    boxes = np.flip(boxes, axis=2)
    boxes = np.reshape(boxes, (100, 8))
    np.savetxt(out_txt_file_path, boxes[:i+1], '%d', ',', ',\n')