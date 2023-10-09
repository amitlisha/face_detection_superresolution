import argparse

args = argparse.ArgumentParser()
args.add_argument("--image-size", type=int, default=736, help="size of image. image must be a square")
args.add_argument("--smart_padding", default=False, action="store_true", help="if true pad the image to the"
                                                                              "max dim of the image")
args.add_argument("--detection_batch_size", type=int, default=20, help="number of frames for inference batch,"
                                                                       "-1 means all the video in one batch")


args.add_argument("--min-face", type=int, default=10, help="minimum possible size of faces in image")
args.add_argument("--visualize-bbs", default=False, action="store_true")
# args.add_argument("--video-path", type=str, help="path to the video to execute face detection on", default=None)
# args.add_argument("--image-path", type=str, help="path to the image to execute face detection on", default=None)
args.add_argument("--scale", type=int, help="Super resolution model scale to use", default=2)
args.add_argument("--save-path", type=str, help="path to where to save output", default="./results")

args.add_argument("--device", type=str, default="cpu")


def parse_args():
    return args.parse_args()
