import face_detection
from utils import image_utils


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

    # img = Image.open("manyfaces.jpg")
    # video, bbs = face_detection.detection_pipeline([img])
    # x = 5

    # video = image_utils.read_images_from_dir("examples")
    video = image_utils.read_video("VID-20231007-WA0155.mp4")[0:500]
    super_draws = []
    for im in video:
        bboxes, _, draws = face_detection.detection_pipeline([im])
        super_draws.append(draws[0])
    image_utils.save_images_to_dir(super_draws, dir_path="video frames with bbs")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
