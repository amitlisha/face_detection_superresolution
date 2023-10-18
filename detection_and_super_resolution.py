import face_detection
from super_resolution import load_upscaler, upscale_crops
from PIL import Image
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


import os
from PIL import Image

def detect_faces_in_video(video):
    """
    Detect faces in the video.
    """
    if config['DEFAULT']['visualize-bbs']:
        video, bbs, _ = face_detection.detection_pipeline(video)
    else:
        video, bbs = face_detection.detection_pipeline(video)
    return video, bbs

def load_super_resolution_model():
    """
    Load the super resolution model.
    """
    model = load_upscaler(config['DEFAULT']['device'], int(config['DEFAULT']['scale']))
    model.eval()
    return model

def process_frame(frame, bb, model):
    """
    Process a single frame: upscale detected faces and return them.
    """
    upscaled_crops = upscale_crops(frame, bb, model, config['DEFAULT']['device'])
    upscaled_images = []
    for j, item in enumerate(upscaled_crops):
        pil_img = Image.fromarray(item)
        image_name = f"face_{j}_scale={config['DEFAULT']['scale']}.jpg"
        upscaled_images.append((image_name, pil_img))
    return upscaled_images

def video_face_detection_and_super_resolution(video):
    """
    Execute super resolution process for a video. Extract and upscale face crops from each frame.
    @param video: a video as a list of PIL images
    @return: a list of PIL images of super resolution faces
    """
    frame_skip_distance = int(config['DEFAULT']['frame-skip-distance'])
    video, bbs = detect_faces_in_video(video)
    model = load_super_resolution_model()
    
    super_resolution_faces = []
    skip_until = -1  # Index until which we should skip processing frames
    
    for i, (frame, bb) in enumerate(zip(video, bbs)):
        if i <= skip_until:
            continue
        if len(bb) > 0:
            upscaled_images = process_frame(frame, bb, model)
            for image_name, pil_img in upscaled_images:
                print(f"upscaled image created: frame_{i}_{image_name}")
                super_resolution_faces.append((f"frame_{i}_{image_name}", pil_img))
            skip_until = i + frame_skip_distance

    return super_resolution_faces
