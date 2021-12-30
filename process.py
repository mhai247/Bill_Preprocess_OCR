import cv2
from pylsd import lsd
import numpy as np
from scipy.spatial import distance as dist
from PIL import Image
import csv

from pyimagesearch import imutils, transform

class Rule():
    def __init__(self, detector, classifier):
        super().__init__()
        self.detector = detector
        self.classifier = classifier
        self.color_dict = {'drug_name': (255, 0, 0), 'usage': (0, 255, 0), 'diagnose': (0, 0, 255), 'type': (255,255,0), 'quantity': (0,255,255), 'date': (255,0,255)}
    
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
        del contours
        del lines
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
        del chosen
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
        del lines
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

    def row(self, img, pts, label):
        x_min = min(pts[:, 0])
        x_max = max(pts[:, 0])
        y_min = min(pts[:, 1])
        y_max = max(pts[:, 1])
        warp = transform.four_point_transform(img, pts)
        ocr = self.predict(warp)
        return (int(x_min), int(y_min), int(x_max), int(y_max), ocr, label)

    def case1(self, img, csv_writer):
        img = self.rotate(img)
        height, width = img.shape[:2]
        h_val = self.check_line(img)
        if len(h_val) < 2:
            return img
        h_min = h_val[-2]- width//50
        h_max = h_val[0] - width//100

        upper = []
        inside_table = []
        dt_boxes, _ = self.detector(img)
        i = 0
        date_done = 0

        for pts in dt_boxes:
            pts = pts.astype(np.int16)
            dist = np.linalg.norm(pts[0] - pts[2])
            if dist > width//30 and pts[0,1] > h_min and pts[0,1] < h_max:
                if date_done == 0:
                    date_box = dt_boxes[i-3]
                    date_done = 1
                inside_table.append(pts)
            elif pts[0,1] < h_val[-1] - height//100:
                upper.append(pts)
            if date_done == 0:
                i += 1
        del dt_boxes
        if date_done == 1:
            self.draw_box(img, date_box, self.color_dict['date'])
            csv_writer.writerow(self.row(img, date_box, 'date'))
        
        phong_kham = upper[0]
        if abs(upper[1][0][1] - phong_kham[0][1]) < height // 100:
            begin = 2
        else:
            begin = 1   
        for i in range (begin, len(upper)):
            csv_writer.writerow(self.row(img, upper[i], 'diagnose'))
            self.draw_box(img, upper[i], self.color_dict['diagnose'])
            if upper[i][0][0] - phong_kham[0][0] < width//70:
                break
        
        del upper

        name_usage = []
        type = []
        quantity = []
        type_start = 0
        max_width = 0
        

        inside_table = sorted(inside_table, key=lambda c: c[0,0])
        # print(inside_table)
        for i in range(1, len(inside_table)):
            if inside_table[i][2][0] - inside_table[i][0][0] > inside_table[max_width][2][0] - inside_table[max_width][0][0]:
                max_width = i


        for pts in inside_table:
            if inside_table[max_width][0][0] - pts[0][0] > width//50:
                continue
            if abs(inside_table[max_width][0][0] - pts[0][0]) < width//50:
                name_usage.append(pts)
            elif type_start == 0:
                type_start = pts[0][0]
                type.append(pts)
                self.draw_box(img, pts, self.color_dict['type'])
            elif pts[0][0] - type_start < width // 50:
                type.append(pts)
                self.draw_box(img, pts, self.color_dict['type'])
            else:
                quantity.append(pts)
                self.draw_box(img, pts, self.color_dict['quantity'])

        del inside_table
        
        type = sorted(type, key=lambda c: c[0][1], reverse=True)
        quantity = sorted(quantity, key=lambda c: c[0][1])

        line_idx = 1
        count = 0

        name_usage = sorted(name_usage, key=lambda c: c[0][1], reverse=True)
        quantity = sorted(quantity, key=lambda c: c[0][1], reverse=True)
        type = sorted(type, key=lambda c: c[0][1], reverse=True)

        quan_idx = 0
        type_idx = 0

        for i in range(len(name_usage)):
            if i == len(name_usage) - 1 or name_usage[i+1][0][1] < h_val[line_idx] - height//100:
                self.draw_box(img, name_usage[i], self.color_dict['drug_name'])
                csv_writer.writerow(self.row(img, name_usage[i], 'drug_name'))
                count = 0
                line_idx += 1
            else:
                if count == 0:
                    csv_writer.writerow(self.row(img, name_usage[i], 'usage'))
                    count = 1
                    self.draw_box(img, name_usage[i], self.color_dict['usage'])
                    if quan_idx < len(quantity) and quantity[quan_idx][0][1] > h_val[line_idx] - height//100:
                        csv_writer.writerow(self.row(img, quantity[quan_idx], 'quantity'))
                        quan_idx += 1
                    if type_idx < len(type) and type[type_idx][0][1] > h_val[line_idx] - height//100:
                        csv_writer.writerow(self.row(img, type[type_idx], 'type'))
                        type_idx += 1
                else:
                    self.draw_box(img, name_usage[i], self.color_dict['drug_name'])
                    csv_writer.writerow(self.row(img, name_usage[i], 'drug_name'))
        del name_usage
        del type
        del quantity
        del h_val
        
        return img

