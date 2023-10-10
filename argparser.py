import argparse

args = argparse.ArgumentParser()
args.add_argument("--detection_batch_size", type=int, default=20, help="number of frames for inference batch,"
                                                                       "-1 means all the video in one batch")
args.add_argument("--min-face", type=int, default=10, help="minimum possible size of faces in image")
args.add_argument("--visualize-bbs", default=True, action="store_true")
args.add_argument("--scale", type=int, help="Super resolution model scale to use, "
                                            "possible values are 2, 3, 4", default=2)
args.add_argument("--save-path", type=str, help="path to where to save output", default="./results")

args.add_argument("--device", type=str, default="cpu")


def parse_args():
    return args.parse_args()
