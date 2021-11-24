import argparse

from config import *
from rotate import rotate
from predict_det import detect
from map import map
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Scan text image',
                        add_help=True)
    parser.add_argument('--image_dir', type=str, default='test',
                        help='Dataset name in data folder')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output step by step')
    return parser.parse_args()


if __name__ == '__main__':
    dataset =  parse_args().image_dir
    start = time.time()
    print('_____________________Rotate phase_____________________\n')
    rotate(dataset)
    print('_____________________Detect phase_____________________\n')
    detect(dataset)
    print('_______________________Map phase______________________\n')
    map(dataset)
    end = time.time()

    print('Total time: {} secs'.format(end - start))