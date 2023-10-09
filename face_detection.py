from face_detector import YoloDetector
import numpy as np
from PIL import Image, ImageDraw
from argparser import parse_args
from utils import image_utils


def detect_faces_in_images(model, img, bs):
    if bs != -1:
        bboxes = []
        list_chunked = [img[i:i + bs] for i in range(0, len(img), bs)]
        for ls in list_chunked:
            bbs, points = model.predict(ls)
            for bb in bbs:
                bboxes.append(bb)
    else:
        bboxes, points = model.predict(img)
    return bboxes


def load_face_detector(target_size=720, device="cpu", min_face=20):
    return YoloDetector(target_size=target_size, device=device, min_face=min_face)


def detection_pipeline(video):
    """
    preprocess the video and returns the processes video and bboxes
    @param video: a list of frames
    @return: process video and a list of lists of bboxes, one for each frame
    """
    args = parse_args()
    model = load_face_detector(target_size=args.image_size, device=args.device, min_face=args.min_face)
    video = image_utils.video_to_images_for_detection(video, args.image_size)
    bboxes = detect_faces_in_images(model, video, args.detection_batch_size)
    if args.visualize_bbs:
        draw_video = image_utils.draw_bbs_on_video(video, bboxes)
        return video, bboxes, draw_video
    return video, bboxes
