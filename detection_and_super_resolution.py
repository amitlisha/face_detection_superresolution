import face_detection
from super_resolution import load_upscaler, upscale_crops
from PIL import Image
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


def video_face_detection_and_super_resolution(video):
    """
    This function executes the super resolution process for a given video. Crops of faces are extracted from each frame
    and are than upsampled with super resolution. The function saves images of the crops in a chosen save path.
    @param video: a video as a list of PIL images
    @return: a list of PIL images of super resolution faces
    """
    # if args.save_path is None:
    #     path = args.video_path if args.video_path else args.image_path
    #     args.save_path = f"results/{os.path.basename(path)}"
    if config['DEFAULT']['visualize-bbs']:
        video, bbs, _ = face_detection.detection_pipeline(video)
        print(bbs)
    else:
        video, bbs = face_detection.detection_pipeline(video)
    model = load_upscaler(config['DEFAULT']['device'], int(config['DEFAULT']['scale']))
    model.eval()

    # os.makedirs(f"{args.save_path}", exist_ok=True)
    super_resolution_faces = []
    for i, (frame, bb) in enumerate(zip(video, bbs)):
        if len(bb) > 0:
            upscaled_crops = upscale_crops(frame, bb, model, config['DEFAULT']['device'])
            for j, item in enumerate(upscaled_crops):
                pil_img = Image.fromarray(item)
                # if args.save_path is not None:
                image_name = f"frame_{i}_face_{j}_scale={config['DEFAULT']['scale']}.jpg"

                print("upscaled image created: " + image_name)

                super_resolution_faces.append((image_name, pil_img))
    return super_resolution_faces