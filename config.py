import os

############################################
ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(ROOT, 'data')
OUTPUT_ROOT = os.path.join(ROOT, 'output')

dataset = '20211015'
############################################


def output_dir(phase, type = None):
    dir = os.path.join(OUTPUT_ROOT, '{}/{}'.format(phase, dataset))

    if type is not None:
        dir = os.path.join(dir, type)

    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            raise PermissionError("Can not make directory")
    return dir

# input_data_dir
in_img_dir = os.path.join(DATA_ROOT, dataset)

# rotate phase
out_rot_img_dir = output_dir('rotate')

# detection phase
out_det_img_dir = output_dir('detect', type='img')
out_det_txt_dir = output_dir('detect', type='txt')

# mapping phase
out_map_img_dir = output_dir('map', type='img')
out_map_txt_dir = output_dir('map', type='txt')
