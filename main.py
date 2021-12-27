from typing import Mapping
from utils import parse_args
from detect import TextDetector
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import glob
import os
import time

from process import Rule
from config import in_img_dir, out_rule_txt_dir, out_rule_img_dir

if __name__ == '__main__':
    args = parse_args()
    detector = TextDetector(args)
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['device'] = 'cpu'
    # config['cnn']['pretrained'] = False
    # config['predictor']['beamsearch'] = False
    classifier = Predictor(config)
    rule = Rule(detector, classifier)

    dataset = args.dataset
    image_dir = in_img_dir(dataset)
    out_txt_dir = out_rule_txt_dir(dataset)
    out_img_dir = out_rule_img_dir(dataset)

    files = glob.glob(image_dir + '/*.jpg')

    for file in reversed(files):
        file_name = file.split('/')[-1]
        img = cv2.imread(file)
        start = time.time()
        name_usage, chandoan, vsl_img = rule.case1(img)
        end = time.time()
        out_txt_file = os.path.join(out_txt_dir, file_name.replace('.jpg', '.txt'))
        out_img_file = os.path.join(out_img_dir, file_name)
        fstream = open(out_txt_file, 'w')

        fstream.write(chandoan + '\n')
        
        for name, usage in name_usage.items():
            fstream.write("\n\n" + name + "\n" + usage)
        
        fstream.close()

        print('File: ' + file)
        print('Output saved to: ' + out_txt_file)
        if args.visualise:
            cv2.imwrite(out_img_file, vsl_img)
            print('Visualised image saved in: ' + out_img_file)
        print('Time: {}\n'.format(end - start))


