"""
Script for performing super resolution on a given image or crops from an image.
"""
import torch
import numpy as np
from .PAN.codes.utils.util import single_forward, tensor2img
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="path to model checkpoint")
    parser.add_argument("--video-path", type=str, help="path to the video to execute super resolution on")
    parser.add_argument("--bs", type=int, help="number of crops to upsample at each forward")
    return parser.parse_args()

def img2tensor(img):
    imgt = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return imgt

def upscale_image(img_batch, model):
    imgt_list = []
    for img in img_batch:
        imgt_list.append(img2tensor(img))
    imgt_batch = torch.tensor(imgt_list)
    output = single_forward(model, imgt_batch.cuda())
    upscaled_images = [tensor2img(output[i]) for i in range(len(output))]
    return upscaled_images


def upscale_crops(img, crops, model, batch_size=-1):
    img_crops = []
    for crop_coords in crops:
        xl, yl, xr, yr = crop_coords
        img_crops.append(img[yl:yr, xl:xr])

    if batch_size > 0:
        upscaled_crops = []
        batch = []
        for img_crop in img_crops:
            batch.append(img_crop)
            if len(batch) == batch_size:
                res = upscale_image(batch, model)
                upscaled_crops += res
                batch = []

        if len(batch) > 0:  # remainder
            res = upscale_image(batch, model)
            upscaled_crops += res
    else:
        return upscale_image(img_crops, model)

if __name__ == '__main__':
    args = parse_args()
    model = torch.load(args.checkpoint)

    #TODO: add face detection function to create complete video analysis function



