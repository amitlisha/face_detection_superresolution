from face_detector import YoloDetector
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import face_detection


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
    img = Image.open("manyfaces.jpg")
    video, bbs = face_detection.detection_pipeline([img])
    x = 5



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
