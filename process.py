import cv2
from pylsd import lsd
import numpy as np
from scipy.spatial import distance as dist
from PIL import Image

from pyimagesearch import imutils, transform

class Rule():
    def __init__(self, detector, classifier):
        super().__init__()
        self.detector = detector
        self.classifier = classifier
    
    def rotate(self, img):
        '''Rotate the image'''
        height, width = img.shape[:2]

        crop_height = int(height/6)
        crop_width = int(width/6)

        crop = img[crop_height:height - crop_height, crop_width:width - crop_width]

        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        crop = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(50,9))
        crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)

        edged = cv2.Canny(crop, 0, 100)

        lines = lsd(edged)

        if lines is not None:
            lines = lines.squeeze().astype(np.int16).tolist()

            horizontal_lines_canvas = np.zeros(edged.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)

        (contours, _) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:1]

        #take longest contour
        contour = contours[0].reshape(contours[0].shape[0], contours[0].shape[2])

        min_x = np.amin(contour[:, 0], axis=0)
        max_x = np.amax(contour[:, 0], axis=0)
        left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
        right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))

        sin = (left_y - right_y)/dist.euclidean((min_x, left_y), (max_x, right_y))
        angle = -np.arcsin(sin) * 180 / np.pi

        if abs(angle) > 25:
            return None

        rotated = imutils.rotate(img, angle)
        return rotated
    
    def check_line(self, img):
        '''Find height coordination of table's horizontal lines'''
        height, width = img.shape[:2]

        crop_width = int(width/6)
        crop = img[:, crop_width:width - crop_width, :]

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(25,1))
        erode = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        edged = cv2.Canny(erode, 10, 20)

        lines = lsd(edged)

        if lines is not None:
            lines = lines.squeeze().astype(np.int16).tolist()

            chosen = []
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) < 45 or y1 < height//4 or y1 > height*3//4:
                    continue
                chosen.append(line)
            lines  = sorted(chosen, key=lambda c: c[1], reverse=True)
        num = 0
        h_val = []
        start = lines[0]
        count = 0
        x_min = min(start[0], start[2])
        x_max = max(start[0], start[2])
        for i in range(len(lines) - 1):
            x1, y1, x2, y2, _ = lines[i]
            if start[1] - y1 <= width/80:
                count += 1
                x_min = min(x_min, x1, x2)
                x_max = max(x_max, x1, x2)
            else:
                if x_max - x_min > width/4:
                    num += 1
                    h_val.append(start[1])
                count = 0
                start = lines[i]
                x_min = min(start[0], start[2])
                x_max = max(start[0], start[2])
        if x_max - x_min > width/2:
            num += 1
            h_val.append(start[1])
        count = 0
        start = lines[i]
        x_min = min(start[0], start[2])
        x_max = max(start[0], start[2])
        return h_val

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        s = self.classifier.predict(im_pil)
        return s

    def draw_box(self, img, pts, color):
        for i in range(4):
            cv2.line(img, tuple(pts[i-1]), tuple(pts[i]), color, thickness=3)
        return

    def case1(self, img):
        img = self.rotate(img)
        height, width = img.shape[:2]
        h_val = self.check_line(img)
        h_min = h_val[-2]- width//50
        h_max = h_val[0] - width//100

        max_width = 0
        upper = []
        inside_table = []
        dt_boxes, _ = self.detector(img)
        for pts in dt_boxes:
            pts = pts.astype(np.int16)
            dist = np.linalg.norm(pts[0] - pts[2])
            if dist > width//25 and pts[0,1] > h_min and pts[0,1] < h_max:
                inside_table.append(pts)
                if pts[2][0] - pts[0][0] > inside_table[max_width][2][0] - inside_table[max_width][0][0]:
                    max_width = len(inside_table) - 1
            elif pts[0,1] < h_val[-1] - height//100:
                upper.append(pts)
        
        name_usage = []
        for pts in inside_table:
            if abs(inside_table[max_width][0][0] - pts[0][0]) < width//50:
                name_usage.append(pts)
        
        line_idx = 1
        count = 0
        name = ''
        usage = ''
        mapping = {}

        color_dict = {'name': (255, 0, 0), 'usage': (0, 255, 0), 'diagnosis': (0, 0, 255)}
        for i in range(len(name_usage)):
            warp = transform.four_point_transform(img, name_usage[i])
            ocr = self.predict(warp)
            if i == len(name_usage) - 1 or name_usage[i+1][0][1] < h_val[line_idx] - height//100:
                self.draw_box(img, name_usage[i], color_dict['name'])
                name = ocr + name
                mapping[name] = usage
                name = ''
                count = 0
                line_idx += 1
            else:
                if count == 0:
                    usage = ocr
                    count = 1
                    self.draw_box(img, name_usage[i], color_dict['usage'])
                else:
                    self.draw_box(img, name_usage[i], color_dict['name'])
                    name = ocr + name
        phong_kham = upper[0][0][0]
        chandoan = ''     
        for i in range (1, len(upper)):
            warp = transform.four_point_transform(img, upper[i])
            ocr = self.predict(warp)
            self.draw_box(img, upper[i], color_dict['diagnosis'])
            chandoan = ocr + ' ' + chandoan
            if upper[i][0][0] - phong_kham < width//70:
                break
        return mapping, chandoan, img

