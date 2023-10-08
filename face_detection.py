from yoloface_master.face_detector import YoloDetector
from argparser import parse_args
from yoloface_master.utils import image_utils
from PIL import Image


def detect_faces_in_images(model, imgs):
    bboxes, points = [], []
    for img in imgs:
        bbox, p = model.predict(img)
        bboxes.append(bbox[0])
        points.append(p[0])
    return bboxes


def load_face_detector(target_size=720, device="cpu", min_face=20):
    return YoloDetector(target_size=target_size, device=device, min_face=min_face)


def detection_pipeline(video, args):
    """
    preprocess the video and returns the processes video and bboxes
    @param video: a list of frames
    @return: process video and a list of lists of bboxes, one for each frame
    """
    model = load_face_detector(target_size=args.image_size, device=args.device, min_face=args.min_face)
    video = image_utils.video_to_images_for_detection(video, args.image_size)
    bboxes = detect_faces_in_images(model, video)
    if args.visualize_bbs:
        draw_video = image_utils.draw_bbs_on_video([Image.fromarray(frame) for frame in video], bboxes)
        return video, bboxes, draw_video
    return video, bboxes
