import cv2
import numpy as np
from numpy.core.fromnumeric import shape
from pyimagesearch import transform
import glob
from config import out_rot_img_dir, out_det_txt_dir, out_rule_txt_dir, out_rule_img_dir, ROOT
import time
import os
from PIL import Image
from pylsd import lsd
import yaml

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
    
def predict(img, detector):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    s = detector.predict(im_pil)
    return s

def check_line(lines, width):
    # print(len(lines))
    # print(width//15)
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
            # print(y1, x_max, x_min)
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

def rule(dataset):
    # config = Cfg.load_config_from_file(os.path.join(ROOT, 'vietocr/config/vgg-seq2seq.yml'))
    with open(os.path.join(ROOT, 'vietocr/config/vgg-seq2seq.yml'), encoding='utf-8') as f:
            config = yaml.safe_load(f)
    with open(os.path.join(ROOT, 'vietocr/config/base.yml'), encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
    base_config.update(config)
    config =  Cfg(base_config)
    config['device'] = 'cpu'
    detector = Predictor(config)
    files = glob.glob(out_rot_img_dir(dataset=dataset) + '/*.jpg')
    for file in reversed(files):
        time_start = time.time()
        img_file_name = file.split('/')[-1]
        txt_file_name = img_file_name.replace('.jpg', '.txt')
        txt_file_path = os.path.join(out_det_txt_dir(dataset), txt_file_name)
        out_txt_file_path = os.path.join(out_rule_txt_dir(dataset), txt_file_name)
        
        out_txt_file = open(out_txt_file_path, 'w')
        txt_file = open(txt_file_path, 'r')
        txt_lines = txt_file.readlines()
        txt_file.close()

        img = cv2.imread(file)
        height, width = img.shape[:2]
        crop_width = int(width/6)

        crop = img[:, crop_width:width - crop_width, :]

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(25,1))
        erode = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        edged = cv2.Canny(erode, 10, 20)

        lines = lsd(edged)
        list_pts = []

        if lines is not None:
            lines = lines.squeeze().astype(np.int32).tolist()

            chosen = []
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) < 45 or y1 < height//4 or y1 > height*3//4:
                    continue
                chosen.append(line)
            lines  = sorted(chosen, key=lambda c: c[1], reverse=True)
            h_val = check_line(lines, width)
            h_min = h_val[-2]- width//50
            h_max = h_val[0] - width//100
            max_width = 0
            upper = []
            for val in h_val:
                cv2.line(img, (0, val), (img.shape[1] - 1, val), (255, 0, 0), 2)
            for txt_line in txt_lines:
                txt_line = txt_line.replace('\n', '')
                idxs = txt_line.split(',')
                pts = []
                for k in range(8):
                    pts.append(int(idxs[k]))

                pts = np.reshape(pts, (4,2))
                dist = np.linalg.norm(pts[0] - pts[2])
                if dist > width//25 and pts[0,1] > h_min and pts[0,1] < h_max:
                    list_pts.append(pts)
                elif pts[0,1] < h_val[-1] - height//100:
                    upper.append(pts)
            
            for i in range(len(list_pts)):
                if list_pts[i][2][0] - list_pts[i][0][0] > list_pts[max_width][2][0] - list_pts[max_width][0][0]:
                    max_width = i
            Name_Usage = []
            for pts in list_pts:
                if abs(list_pts[max_width][0][0] - pts[0][0]) < width//50:
                    Name_Usage.append(pts)
            
            line_idx = 1
            count = 0
            name = ''
            usage = ''
            mapping = {}
            for i in range(len(Name_Usage)):
                warp = transform.four_point_transform(img, Name_Usage[i])
                ocr = predict(warp, detector)
                if i == len(Name_Usage) - 1 or Name_Usage[i+1][0][1] < h_val[line_idx] - height//100:
                    name = ocr + name
                    mapping[name] = usage
                    name = ''
                    count = 0
                    line_idx += 1
                else:
                    if count == 0:
                        usage = ocr
                        count = 1
                    else:
                        name = ocr + name
            phong_kham = upper[0][0][0]
            chandoan = ''     
            for i in range (1, len(upper)):
                warp = transform.four_point_transform(img, upper[i])
                ocr = predict(warp, detector)
                chandoan = ocr + ' ' + chandoan
                if upper[i][0][0] - phong_kham < width//70:
                    break
            
            out_txt_file.write(chandoan + '\n')
            for name, usage in mapping.items():
                out_txt_file.write('\n\n' + name + '\n' + usage)

        out_txt_file.close()                

        print('FILE: {}'.format(img_file_name))
        print("Time: {}".format(time.time() - time_start))
        print('Save output to {}\n'.format(out_txt_file_path))

if __name__ == '__main__':
    dataset = '20211015'
    rule(dataset)
