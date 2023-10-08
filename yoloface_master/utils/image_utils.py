import cv2
from PIL import ImageDraw
import numpy as np


def resize_image(img, size: int):
    return cv2.resize(np.array(img), (size, size), interpolation=cv2.INTER_LINEAR)


def video_to_images_for_detection(video, size):
    new_video = []
    for frame in video:
        frame = np.array(resize_image(frame, size))
        new_video.append(frame)
    return new_video


def draw_bbs_on_video(video, bboxes):
    draw_frames = []
    for frame, bbs in zip(video, bboxes):
        draw_frame = draw_bbs_on_image(frame, bbs)
        draw_frames.append(draw_frame)
    return draw_frames


def draw_bbs_on_image(img, bbs: list):
    image_to_draw = ImageDraw.Draw(img)
    for bb in bbs:
        start = (bb[0], bb[1])
        end = (bb[2], bb[3])
        image_to_draw.rectangle([start, end], outline="red")
    return img
