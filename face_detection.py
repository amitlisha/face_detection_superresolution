from face_detector import YoloDetector
import numpy as np
from PIL import Image, ImageDraw
from argparser import parse_args
from utils import image_utils


def detect_faces_in_images(model, img):
    bboxes, points = model.predict(img)
    return bboxes


def load_face_detector(target_size=720, device="cpu", min_face=20):
    return YoloDetector(target_size=target_size, device=device, min_face=min_face)


def detection_pipeline(video):
    args = parse_args()
    model = load_face_detector(target_size=args.target_size, device=args.device, min_face=args.min_face)
    video = image_utils.video_to_images_for_detection(video, args.size)
    bboxes = detect_faces_in_images(model, video)
    if args.visualize_bbs:
        draw_video = image_utils.draw_bbs_on_video(video, bboxes)
        return video, bboxes, draw_video
    return video, bboxes
