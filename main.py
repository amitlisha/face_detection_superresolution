import io

import numpy as np
from s3 import s3, bucket_name
from argparser import parse_args
from image_utils import read_video, read_image, get_temporary_file_name
from detection_and_super_resolution import video_face_detection_and_super_resolution
from fastapi import FastAPI, UploadFile, HTTPException
from typing import Annotated

TEMP_FILE_LOCATION = './input-images/temp'

args = parse_args()
app = FastAPI()


@app.post("/super-resolution")
def super_resolution(file: UploadFile):
    if file is None:
        raise HTTPException(status_code=400, detail="a file was not provided")
    file_name = get_temporary_file_name(file.file)
    video = read_video(file_name)
    super_resolution_faces = video_face_detection_and_super_resolution(video, args)
    for image in super_resolution_faces:
        image_byte_arr = io.BytesIO()
        image[1].save(image_byte_arr, format='JPEG')
        image_byte_arr = image_byte_arr.getvalue()
        s3.put_object(Key=image[0], Body=image_byte_arr, Bucket=bucket_name)
        print("image uploaded to s3: " + image[0])
    print(s3.list_objects(Bucket=bucket_name))
    return ''

# args = parse_args()
# if args.video_path is None and args.image_path is None:
#     raise IOError("no input path was given, please give image_path or video_path")
# if args.video_path and args.image_path:
#     raise IOError("video_path or image_path most be None")
# if args.video_path:
#     video = read_video(args.video_path)
# else:
#     video = read_image(args.image_path)
# super_resolution_faces = video_face_detection_and_super_resolution(video, args)
# print(super_resolution_faces)
