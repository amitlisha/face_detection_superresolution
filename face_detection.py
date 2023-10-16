from yoloface_master.face_detector import YoloDetector
from argparser import parse_args
from yoloface_master.utils import image_utils
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


def detect_faces_in_images(model, img, bs):
    """
    detect faces in batches
    @param model: the detection model
    @param img: the list of images
    @param bs: batch size for inference
    @return: bboxes
    """
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
    """
    create face detector model
    @param target_size: images size for detection
    @param device: device for detector
    @param min_face: minimum possible size of faces in images in pixels
    @return: detector model
    """
    return YoloDetector(target_size=target_size, device=device, min_face=min_face)


def detection_pipeline(video):
    """
    preprocess the video and returns the processes video and bboxes
    @param video: a list of frames
    @return: process video and a list of lists of bboxes, one for each frame
    """
    #args = parse_args()
    frame = video[0]
    image_size = max(frame.size)
    model = load_face_detector(target_size=image_size, device=config['DEFAULT']['device'], min_face=int(config['DEFAULT']['min-face']))
    video = image_utils.video_to_images_for_detection(video)
    bboxes = detect_faces_in_images(model, video, int(config['DEFAULT']['detection_batch_size']))
    if config['DEFAULT']['visualize-bbs']:
        draw_video = image_utils.draw_bbs_on_video(video, bboxes)
        return video, bboxes, draw_video
    return video, bboxes
