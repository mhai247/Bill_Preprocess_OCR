from utils import parse_args
from detect import TextDetector
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import glob
import os
import time
import csv

from process import Rule
from config import in_img_dir, out_rule_csv_dir, out_rule_img_dir

if __name__ == '__main__':
    args = parse_args()
    detector = TextDetector(args)
    config = Cfg.load_config_from_name('vgg_seq2seq')
    if args.use_gpu == True:
        config['device'] = 'cuda'
    else:
        config['device'] = 'cpu'
    # config['cnn']['pretrained'] = False
    # config['predictor']['beamsearch'] = False
    classifier = Predictor(config)
    rule = Rule(detector, classifier)

    dataset = args.dataset
    image_dir = in_img_dir(dataset)
    out_csv_dir = out_rule_csv_dir(dataset)
    out_img_dir = out_rule_img_dir(dataset)

    files = glob.glob(os.path.join(image_dir, '*.jpg'))
    # print(files)

    for file in reversed(files):
        file_name = file.split('/')[-1]
        out_csv_file = os.path.join(out_csv_dir, file_name.replace('.jpg', '.csv'))
        out_img_file = os.path.join(out_img_dir, file_name)
        fstream = open(out_csv_file, 'w')
        writer = csv.writer(fstream)

        img = cv2.imread(file)
        start = time.time()
        vsl_img = rule.case1(img, writer)
        end = time.time()
       
        
        fstream.close()

        print('File: ' + file)
        print('Output saved to: ' + out_csv_file)
        if args.visualise:
            cv2.imwrite(out_img_file, vsl_img)
            print('Visualised image saved in: ' + out_img_file)
        print('Time: {}\n'.format(end - start))
        # break


