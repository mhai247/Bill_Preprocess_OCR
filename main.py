from typing import Mapping
from utils import parse_args
from detect import TextDetector
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import glob
import os

from process import Rule
from config import in_img_dir, out_rule_txt_dir

if __name__ == '__main__':
    args = parse_args()
    detector = TextDetector(args)
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['device'] = 'cpu'
    # config['cnn']['pretrained'] = False
    # config['predictor']['beamsearch'] = False
    classifier = Predictor(config)
    rule = Rule(args, detector, classifier)

    dataset = args.dataset
    image_dir = in_img_dir(dataset)
    out_dir = out_rule_txt_dir(dataset)

    files = glob.glob(image_dir + '/*.jpg')

    for file in files:
        file_name = file.split('/')[-1]
        img = cv2.imread(file)
        name_usage, chandoan = rule(img)
        out_file = os.path.join(out_dir, file_name.replace('.jpg', '.txt'))
        fstream = open(out_file, 'w')

        fstream.write(chandoan + '\n')
        
        for name, usage in name_usage.items():
            fstream.write("\n\n" + name + "\n" + usage)
        
        fstream.close()


