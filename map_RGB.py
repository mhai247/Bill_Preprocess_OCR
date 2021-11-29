import cv2
import numpy as np
from numpy.core.fromnumeric import shape
from pyimagesearch import transform
import glob
from config import out_rot_img_dir, out_det_txt_dir, out_map_img_rgb_dir, out_map_txt_dir
import time
import os

def overlap(box1, box2):
    if max(box1[:, 1]) - min(box2[:, 1]) <= 15:
        return 0
    if min(box1[:, 1]) - max(box2[:, 1]) >= -15:
        return 0
    if max(box1[:, 0]) <= min(box2[:, 0]):
        return 0
    if min(box1[:, 0]) >= max(box2[:, 0]):
        return 0
    return 1


def trim(img, origin_img):
    h,w = img.shape

    horizon_dim = np.amin(img, axis=0)
    text_appear = np.where(horizon_dim == 0)

    horizon_min = 0
    horizon_max = w

    if len(text_appear[0]) == 0:
        return origin_img[0:h, 0:w, :],0

    elif len(text_appear[0]) > 5:
        horizon_min = max(0, np.min(text_appear)-1)
        horizon_max = min(w, np.max(text_appear)+1)

    blank = np.where(horizon_dim[:horizon_max] != 0)
    i = 0
    if len(blank[0]) >= 15:
        last = np.max(blank[0])
        for i in range(len(blank[0])):
            if blank[0][-i-1] != last - i:
                break
        
        if i > 15:
            horizon_max -= i

    if len(blank[0][horizon_min:]) >= 15:
        first = np.min(blank[0][horizon_min:])
        for j in range(len(blank[0][horizon_min:])):
            if blank[0][j] != first + j:
                break
        
        if j > 15:
            horizon_min += j - 1

    vertical_dim = np.amin(img, axis=1)
    text_appear = np.where(vertical_dim == 0)
    vertical_min = max(0, np.min(text_appear)-1)
    vertical_max = min(h, np.max(text_appear)+1)

    return origin_img[vertical_min:vertical_max, horizon_min:horizon_max, :],i


def map(dataset):
    files = glob.glob(out_rot_img_dir(dataset=dataset) + '/*.jpg')
    for file in reversed(files):
        # print(file)
        img_file_name = file.split('/')[-1]
        txt_file_name = img_file_name.replace('.jpg', '.txt')
        
        img = cv2.imread(file)
        txt_file_path = os.path.join(out_det_txt_dir(dataset), txt_file_name)
        out_img_file_path = os.path.join(out_map_img_rgb_dir(dataset), img_file_name)
        out_txt_file_path = os.path.join(out_map_txt_dir(dataset), txt_file_name)

        txt_file = open(txt_file_path, 'r')

        start = time.time()

        height, width = img.shape[:2]
        blank_image = np.ones((height,width, 3), np.uint8) * 255

        MAX_DIS = height/80

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
            # print(pts)
            # cv2.imshow('sharpen', warp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            h,w = warp.shape[:2]

            gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)

            # sharpen image
            sharpen = cv2.GaussianBlur(gray, (0,0), 3)
            sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

            # apply adaptive threshold to get black and white effect
            thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 9)


            tl = pts[0]
            tr = pts[1]

            if tl[1] + h >= height:
                continue
            if tr[0] - w < 0:
                continue

            thresh, dis = trim(thresh, warp)
            h,w = thresh.shape[:2]

            
            # cv2.imshow('sharpen', cv2.resize(thresh, (w,h)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 0 for left side, 1 for right
            side = 0

            if tr[0] >= width/2:
                side = 1
                new_box = np.array( [[tl[1], tl[0]], [tl[1],tl[0] + w], [tl[1] + h, tl[0]+w], [tl[1] + h, tl[0]]])
            else:
                side = 0
                tr[0] = tr[0] - dis
                new_box = np.array( [[tr[1], tr[0]-w], [tr[1], tr[0]], [tr[1]+h, tr[0]], [tr[1] + h, tr[0] - w]])

            if side == 1:
                top = tl[1]
            else: top = tr[1]

            #align
            if i > 0 and abs(boxes[i-1][0][0] - top) < MAX_DIS:
                do = 1
                add = boxes[i-1][0][0] - top

                if side == 1:
                    change_box = np.array( [[add + top, tl[0]], [add + top,tl[0] + w], [add + top + h, tl[0]+w], [add + top + h, tl[0]]])
                else:
                    change_box = np.array( [[add + top, tr[0] - w], [add + top, tr[0]], [add + top + h, tr[0]], [add + top + h, tr[0] - w]])
                for j in range(min(i-1, 5)):
                    if overlap(boxes[i-j-1], change_box):
                        do = 0
                        break
                if do:
                    top = boxes[i-1][0][0]

            #prevent overlaping
            max_h = top
            for j in range(min(i-1, 5)):
                if overlap(boxes[i-j-1], new_box):
                    if boxes[i-j-1][3][0] > max_h:
                        max_h = boxes[i-j-1][3][0]

            top = max_h


            if side == 0 and tl[0] + w >= width:
                # out = tl[0] + h - width + 1
                tl[0] = width - w - 1
            # print(blank_image.shape)

            dilate = 0

            if side == 0 and tr[0] < w:
                tr[0] = w
                
            if side == 1:
                blank_image[top+dilate:top+h-dilate, tl[0]+dilate:tl[0]+w- dilate, :] = thresh[dilate:h-dilate, dilate:w-dilate, :]
                boxes[i] = np.array([[top, tl[0]], [top, tl[0] + w], [top + h, tl[0] + w], [top + h, tl[0]]])
            else:
                blank_image[top+dilate:top+h-dilate, tr[0]-w+dilate:tr[0]- dilate, :] = thresh[dilate:h-dilate, dilate:w-dilate, :]
                boxes[i] = np.array([[top, tr[0] - w], [top, tr[0]], [top + h, tr[0]], [top + h, tr[0] - w]])
            # blank_image = cv2.rectangle(blank_image, (boxes[i][0][1], boxes[i][0][0]), (boxes[i][2][1], boxes[i][2][0]), color=(0,0,0), thickness=2)

            # if img_file_name == 'z2848587299374_cfdddfeed125784c6eaed83e83ebaaa5.jpg':
            # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            
            # cv2.imshow('output', blank_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print(h,w,top, tl[0])

        end = time.time()

        cv2.imwrite(out_img_file_path, blank_image)
        # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        # cv2.imshow('output', cv2.resize(blank_image, ())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print("Saved image in {}".format(out_img_file_path))
        print("Time: {}\n".format(end-start))
        boxes = np.flip(boxes, axis=2)
        boxes = np.reshape(boxes, (100, 8))
        np.savetxt(out_txt_file_path, boxes[:i+1], '%d', ',', ',\n')


if __name__ == '__main__':
    dataset = '20211015'
    map(dataset)