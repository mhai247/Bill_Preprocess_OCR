import argparse

def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--visualise", type=str2bool, default=False)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--dataset", type=str)
    return parser.parse_args()