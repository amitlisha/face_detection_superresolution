import argparse

args = argparse.ArgumentParser()
args.add_argument("--image_size", type=int, default=720, help="size of image. image most be a square")
args.add_argument("--device", type=str, default="cpu")
args.add_argument("--min_face", type=int, default=10, help="minimum possible size of faces in image")
args.add_argument("--visualize_bbs", type=bool, default=False, action="store true")


def parse_args():
    return args.parse_args()
