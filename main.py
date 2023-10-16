import io
import numpy as np
from typing import List, Tuple
from PIL import Image
from s3 import s3, bucket_name
from argparser import parse_args
from yoloface_master.utils.image_utils import read_video, read_image, get_temporary_file_name
from detection_and_super_resolution import video_face_detection_and_super_resolution
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Annotated


app = FastAPI()

def process_and_upload_to_s3(data: List[Tuple[str, Image.Image]]):
    for name, img in data:
        image_byte_arr = io.BytesIO()
        img.save(image_byte_arr, format='JPEG')
        image_byte_arr = image_byte_arr.getvalue()
        s3.put_object(Key=name, Body=image_byte_arr, Bucket=bucket_name)
        print(f"Image uploaded to s3: {name}")

    print(s3.list_objects(Bucket=bucket_name))


@app.post("/super-resolution-video")
def super_resolution_video(file: UploadFile):
    if file is None:
        raise HTTPException(status_code=400, detail="a file was not provided")

    file_name = get_temporary_file_name(file.file)
    video = read_video(file_name)
    super_resolution_faces = video_face_detection_and_super_resolution(video)
    process_and_upload_to_s3(super_resolution_faces)
    return ''


@app.post("/super-resolution-image")
def super_resolution_image(file: UploadFile):
    if file is None:
        raise HTTPException(status_code=400, detail="a file was not provided")

    file_name = get_temporary_file_name(file.file)
    image = read_image(file_name)
    super_resolution_faces = video_face_detection_and_super_resolution(image)
    process_and_upload_to_s3(super_resolution_faces)
    return ''
