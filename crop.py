from PIL.ImageFont import ImageFont
import cv2
import numpy as np
from numpy.core.fromnumeric import shape
from pyimagesearch import transform
import glob
from config import out_rot_img_dir, out_det_txt_dir, out_crop_txt_dir, out_crop_img_dir, ROOT
import time
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def predict(img, detector):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    s = detector.predict(img)
    return s

def crop(dataset):
    config = Cfg.load_config_from_file(os.path.join(ROOT, 'vietocr/config/vgg-seq2seq.yml'))
    config['device'] = 'cpu'
    detector = Predictor(config)
    files = glob.glob(out_rot_img_dir(dataset=dataset) + '/*.jpg')
    for file in files:
        # print(file)
        img_file_name = file.split('/')[-1]
        txt_file_name = img_file_name.replace('.jpg', '.txt')
        
        img = cv2.imread(file)
        txt_file_path = os.path.join(out_det_txt_dir(dataset), txt_file_name)
        out_img_file_path = os.path.join(out_crop_img_dir(dataset), img_file_name)
        out_txt_file_path = os.path.join(out_crop_txt_dir(dataset), txt_file_name)
        
        out_txt_file = open(out_txt_file_path, 'w')
        txt_file = open(txt_file_path, 'r')

        start = time.time()

        height, width = img.shape[:2]
        # blank_image = Image.new ( "RGB", (width,height), (255,255,255) )

        line = []
        list_pts = []
        lines = txt_file.readlines()
        txt_file.close()
        
        for line in lines:
            line = line.replace('\n', '')
            idxs = line.split(',')
            pts = []
            for k in range(8):
                pts.append(int(idxs[k]))

            pts = np.reshape(pts, (4,2))
            dist = np.linalg.norm(pts[0] - pts[2])
            if dist > 70:
                list_pts.append(pts)

        arr_pts = np.array(list_pts)

        padding = 50
        min_w = np.min(arr_pts[:,0,0])
        left_margin = max(min_w - padding, 0)

        max_w = np.max(arr_pts[:,1,0])

        right_margin = min(max_w + padding, width)

        min_h = np.min(arr_pts[:,0,1])
        top_margin = max(0, min_h - padding)

        max_h = np.max(arr_pts[:,2,1])
        bottom_margin = min(max_h + padding, height)
        # print(left_margin, right_margin, top_margin, bottom_margin)
        crop = img[top_margin:bottom_margin, left_margin:right_margin,:]
        
        img_RGB = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        plt_img = Image.fromarray(img_RGB)
        draw = ImageDraw.Draw(plt_img)
        font_size = int(height/130)
        unicode_font = ImageFont.truetype("DejaVuSans.ttf", font_size)

        for line in lines:
            line = line.replace('\n', '')
            idxs = line.split(',')
            pts = []
            for k in range(8):
                pts.append(int(idxs[k]))

            pts = np.reshape(pts, (4,2))

            warp = transform.four_point_transform(img, pts)

            ocr = predict(warp, detector)

            pts[:, 1] -= top_margin
            pts[:, 0] -= left_margin
            coor = ''
            # flat = np.reshape(pts, 8)
            
            for val in pts:
                coor += '{},{},'.format(val[0], val[1])

            
            # print(pts.shape)
            if np.min(pts) >= 0:
                out_txt_file.write(coor + ocr + '\n')
                draw.text((pts[3][0], pts[0][1]), ocr, font=unicode_font, fill=(0,255,0))
                draw.line(pts, fill="red", width=3)
                for point in pts:
                    draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill="red")

        end = time.time()
        plt_img.save(out_img_file_path)
        print('FILE: {}'.format(img_file_name))
        print('Save output to {}'.format(out_img_file_path))
        print("Time: {}\n".format(end-start))

if __name__ == '__main__':
    dataset = '20211015'
    crop(dataset)