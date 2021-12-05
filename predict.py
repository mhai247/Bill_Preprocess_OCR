import argparse
from PIL import Image
from glob import glob
import os
import cv2

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from config import out_map_img_rgb_dir, out_map_txt_dir, out_cls_txt_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--config', required=True, help='foo help')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    detector = Predictor(config)

    img = Image.open(args.img)
    s = detector.predict(img)

    print(s)

def predict(dataset):
    files = glob(out_map_txt_dir(dataset) + '/*.txt')
    for file in files:
        txt_file_name = file.split('/')[-1]
        img_file_name = txt_file_name.replace('.txt', '.jpg')
        img_file_path = os.path.join(out_map_img_rgb_dir(dataset), img_file_name)
        img = Image.open(img_file_path)
        print(type(img))
        
        txt_file = open(file, 'r')
        lines = txt_file.readlines()
        txt_file.close()
        # print(lines)
        break

if __name__ == '__main__':
    predict('20211015')
