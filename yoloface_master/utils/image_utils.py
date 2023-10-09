import os

import cv2
from PIL import ImageDraw, Image
import numpy as np


def read_video(video_path):
    vidObj = cv2.VideoCapture(video_path)
    success = 1
    frames = []
    while success:
        success, image = vidObj.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(image))
    return frames


def read_images_from_dir(dir_path):
    images_names = os.listdir(dir_path)
    images_ls = []
    for image in images_names:
        images_ls.append(Image.open(f"{dir_path}/{image}"))
    return images_ls


def save_images_to_dir(image_ls, dir_path="bb_examples"):
    os.makedirs(dir_path, exist_ok=True)
    for i, image in enumerate(image_ls):
        image.save(f"{dir_path}/{i+1}.png")


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
        if type(frame) != Image:
            frame = Image.fromarray(frame)
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
