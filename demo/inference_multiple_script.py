import os
import sys
import skimage
import hashlib
import requests
import argparse
import skimage.io
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from predictor_carpart import COCODemo
from maskrcnn_benchmark.config import cfg

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]    # convert to BGR format
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

def get_config(config_path):
    config_file = config_path   # update the config options with the config file
    cfg.merge_from_file(config_file)    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    coco_demo = COCODemo(
        cfg#,
        #min_image_size=800,
        #confidence_threshold=0.7,
    )
    return coco_demo

def predict_multiple_image(coco_demo, image_path, folder_path):
    # print('predict multiple image')
    # print('image_path')
    # print(image_path)
    # print(os.listdir(image_path))
    for i in range(len(os.listdir(image_path))):
        # print('image_path')

        image_name = os.listdir(image_path)[i]
        full_image_path = image_path + os.listdir(image_path)[i]

        print('full_image_path')
        print(full_image_path)
        image = skimage.io.imread(full_image_path)

        predictions = coco_demo.run_on_opencv_image(image)  # compute predictions    skimage.io.imsave('image.jpg', predictions)

        image_location = hashlib.sha256(str(image_name).encode('utf-8')).hexdigest()
        full_image_location = folder_path + image_location + '.jpg'
        print('full_image_location')
        print(full_image_location)
        skimage.io.imsave(full_image_location, predictions)
        print('saved')
        imshow(predictions)
    # plt.show()

def main():
    """
    Inference script
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--config_path', help="yml file for inference")
    parser.add_argument('-i', '--image_path', help="multiple image path")
    parser.add_argument('-f', '--folder_path', help="saving image to specific folder")
    args = parser.parse_args()
    # config_path = '../configs/test_inference.yaml'
    # image_path = '/home/sulabh/Pictures/Damage/Final/fb/bumper-dent-repair-dallas-tx-18-410x300.jpg'
    coco_demo = get_config(args.config_path)
    predict_multiple_image(coco_demo, args.image_path, args.folder_path)

if __name__=="__main__":
    main()