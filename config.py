import os

############################################
ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(ROOT, 'data')
OUTPUT_ROOT = os.path.join(ROOT, 'output')
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

# input_data_dir
def in_img_dir(dataset='test'):
    dir = os.path.join(DATA_ROOT, dataset)
    return dir

def out_rule_txt_dir(dataset='rule'):
    out_rule_txt_dir = output_dir(dataset, 'rule', type='txt')
    return out_rule_txt_dir