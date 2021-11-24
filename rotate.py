from pyimagesearch import imutils
from scipy.spatial import distance as dist
import numpy as np
import cv2
from pylsd import lsd
import time
from config import in_img_dir,out_rot_img_dir
import glob
import os


def rotate(dataset):
    files = glob.glob(in_img_dir(dataset)+'/*.jpg')

    for file in files:
        start = time.time()
        img_name = file.split('/')[-1]
        img = cv2.imread(file)
        height, width = img.shape[:2]

        crop_height = int(height/6)
        crop_width = int(width/6)

        crop = img[crop_height:height - crop_height, crop_width:width - crop_width]

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(50,9))
        dilated = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        edged = cv2.Canny(dilated, 0, 100)

        lines = lsd(edged)

        corners = []
        if lines is not None:
            lines = lines.squeeze().astype(np.int32).tolist()

            horizontal_lines_canvas = np.zeros(edged.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)

        (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # new_height, new_width = edged.shape
        # blank_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        #take longest contour
        contour = contours[0].reshape(contours[0].shape[0], contours[0].shape[2])

        min_x = np.amin(contour[:, 0], axis=0)
        max_x = np.amax(contour[:, 0], axis=0)
        left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
        right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))

        sin = (left_y - right_y)/dist.euclidean((min_x, left_y), (max_x, right_y))
        angle = -np.arcsin(sin) * 180 / np.pi

        rotated = imutils.rotate(img, angle)
        # cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
        # cv2.imshow('output', none)
        # cv2.imshow('origin', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        out_file_path = os.path.join(out_rot_img_dir(dataset), img_name)
        cv2.imwrite(out_file_path, rotated)

        print(file + '\tTime: {} secs'.format(time.time() - start))

if __name__ == '__main__':
    dataset = 'test'
    rotate(dataset)