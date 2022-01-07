from config import ROOT
import argparse
import os

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--visualise", type=str2bool, default=False)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=500)

    # params for text detector
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str, default= os.path.join(ROOT, 'PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer'))
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.2)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--max_batch_size", type=int, default=10)
    
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    return parser.parse_args()