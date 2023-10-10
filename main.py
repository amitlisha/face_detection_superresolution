from argparser import parse_args
from image_utils import read_video
from detection_and_super_resolution import video_face_detection_and_super_resolution

args = parse_args()
video = read_video(args.video_path)
super_resolution_faces = video_face_detection_and_super_resolution(video, args)
