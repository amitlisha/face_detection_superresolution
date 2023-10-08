from PIL import Image

import face_detection
import super_resolution
from argparser import parse_args
import numpy as np
from skvideo.io import vread
import cv2

if __name__ == '__main__':
    # model = YoloDetector(target_size=720, device="cpu", min_face=20)
    # img = Image.open("manyfaces.jpg")
    # real_image = img.resize((720, 720))
    # img = np.array(real_image)
    # bboxes, _ = detect_faces_in_image(model, img)
    # image_to_draw = ImageDraw.Draw(real_image)
    # for bb in bboxes[0]:
    #     start = (bb[0], bb[1])
    #     end = (bb[2], bb[3])
    #     image_to_draw.rectangle([start, end], outline="red")
    # real_image.show()
    args = parse_args()
    if args.device.isnumeric():
        args.device = int(args.device)

    if args.video_path is not None and args.image_path is not None:
        raise ValueError("Both video_path and image_path arguments are set, only one should be passed")
    elif args.video_path is not None:
        video = vread(args.video_path)
    elif args.image_path is not None:
        video = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Both video_path and image_path arguments are not set, one should be passed")

    super_resolution.face_super_resolution(video, args)




