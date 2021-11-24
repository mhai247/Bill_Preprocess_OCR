import os

############################################
ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(ROOT, 'data')
OUTPUT_ROOT = os.path.join(ROOT, 'output')
MODEL_DIR = os.path.join(ROOT, 'PaddleOCR/models')
############################################


def output_dir(dataset, phase, type=None):
    dir = os.path.join(OUTPUT_ROOT, '{}/{}'.format(phase, dataset))

    if type is not None:
        dir = os.path.join(dir, type)

    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            raise PermissionError("Can not make directory")
    return dir

# det model dir
det_model_dir = os.path.join(ROOT, 'PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer')


# input_data_dir
def in_img_dir(dataset='test'):
    dir = os.path.join(DATA_ROOT, dataset)
    return dir


# rotate phase
def out_rot_img_dir(dataset='test'):
    out_rot_img_dir = output_dir(dataset, 'rotate')
    return out_rot_img_dir


# detection phase
def out_det_img_dir(dataset='test'):
    out_det_img_dir = output_dir(dataset, 'detect', type='img')
    return out_det_img_dir


def out_det_txt_dir(dataset='test'):
    out_det_txt_dir = output_dir(dataset, 'detect', type='txt')
    return out_det_txt_dir


# mapping phase
def out_map_txt_dir(dataset='test'):
    out_map_txt_dir = output_dir(dataset, 'map', type='txt')
    return out_map_txt_dir


def out_map_img_dir(dataset='test'):
    out_map_img_dir = output_dir(dataset, 'map', type='img')
    return out_map_img_dir