"""
Script for performing super resolution on a given image or crops from an image.
"""
import torch
import numpy as np

import face_detection
from PAN.codes.models.archs import PAN_arch
from PAN.codes.utils.util import single_forward, tensor2img
from PIL import Image
import os
import cv2

def load_upscaler(device="cpu", scale=2):
    model = PAN_arch.PAN(in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=scale)
    if scale == 2:
        model_weight = torch.load('./PAN/experiments/pretrained_models/PANx2_DF2K.pth')
    elif scale == 3:
        model_weight = torch.load('./PAN/experiments/pretrained_models/PANx3_DF2K.pth')
    elif scale == 4:
        model_weight = torch.load('./PAN/experiments/pretrained_models/PANx4_DF2K.pth')
    else:
        raise ValueError(f"no model with scale {scale} exists")
    model.load_state_dict(model_weight)
    model = model.to(device)
    return model


def img2tensor(img):
    imgt = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return imgt

def upscale_image(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32) / 255
    imgt = img2tensor(img)
    imgt = imgt[None, ...]

    output = single_forward(model, imgt.cuda())
    return tensor2img(output)


def upscale_crops(img, crops, model):
    img_crops = []

    for crop_coords in crops:
        xl, yl, xr, yr = crop_coords
        xl = max(0, xl - 25)
        xr = min(img.shape[1], xr + 25)
        yl = max(0, yl - 25)
        yr = min(img.shape[0], yr + 25)
        img_crops.append(img[yl:yr, xl:xr])

    upscaled_crops = []
    for crop in img_crops:
        output = upscale_image(crop, model)
        upscaled_crops.append(output)

    return [item for item in upscaled_crops]


def face_super_resolution(video, args):
    if args.visualize_bbs:
        video, bbs, _ = face_detection.detection_pipeline(video)
    else:
        video, bbs = face_detection.detection_pipeline(video)
    model = load_upscaler(args.device, args.scale)
    model.eval()

    os.makedirs("./results", exist_ok=True)
    for i, (frame, bb) in enumerate(zip(video, bbs)):
        if len(bb) > 0:
            upscaled_crops = upscale_crops(frame, bb, model)
            os.makedirs(f"./results/frame_{i}", exist_ok=True)
            for j, item in enumerate(upscaled_crops):
                pil_img = Image.fromarray(item)
                pil_img.save(f"{args.save_path}/frame_{i}/face_{j}_scale={args.scale}.jpg")