# # Managing Paths and Libraries
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

import json
import hashlib
import skimage
import argparse
import skimage.io
import numpy as np
import skimage.draw

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.draw import polygon, polygon_perimeter
import matplotlib.pyplot as plt

import argparse
import skimage.io
import numpy as np


import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
from skimage.transform import rescale, resize

import pickle

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def make_directory(folder_path):
    """
    Makes directory to save the text file if the path
    doesn't exists
    :param folder_path:
    :return: None
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

def load_raw_json_data(raw_json):
    """
    Loads the raw json data
    :param raw_json: The json which contains the image url
    :return: json_data
    """
    with open(raw_json) as f:
        json_data = json.load(f)
    return json_data

def category_selector(features):
    """
    gives the present category to save
    :param features: to check the name of the category
    :return: category that is choosen
    """
    if features['category'] == 'dent':
        category_now = 'dent'
    elif features['category'] == 'scratch':
        category_now = 'scratch'
    elif features['category'] == 'gone':
        category_now = 'gone'
    elif features['category'] == 'crack':
        category_now = 'crack'
    elif features['category'] == 'bumper':
        category_now = 'bumper'
    else:
        category_now = 'others'
    return category_now

def save_image(image, folder_path, count, key, draw, x, y, category_now, i,
               main_dict, image_location, coco_demo):
    """
    Saves the image in folder
    :param image: The image to save
    :param folder_path: Folder to save
    :param count: For unique id
    :param key: For unique id
    :param draw: True for masking, False for no-labels
    :param x: x-vertices
    :param y: y-vertices
    :param category_now: category of different damages
    :param i: For unique id
    :param unet_labels: True if masking is done for unet, False otherwise
    :return: None
    """
    print('inside save image')
    fnt = ImageFont.truetype('../font/FreeMono.ttf', 40)
    img_pil = Image.fromarray(image)

    drw = ImageDraw.Draw(img_pil, 'RGBA')
    drw.polygon(list(zip(x, y)), outline='red', fill = (255, 255, 255, 127))
    drw.text((x[0], y[0]), category_now, font=fnt, fill=(255, 255, 255, 128))
    # # drw.text((x[0], y[0] + 20), str(r['scores'][i]), font=fnt, fill=(255, 255, 255, 128))

    # make_directory('{0}/{1}/'.format(folder_path, category_now))

    # img_pil.save('{0}/{1}/{2}.png'.format(folder_path, category_now,
    #                                       hashlib.md5(bytes(int(str(count) + str(i)))).hexdigest()), 'PNG')

    # # Now detecting and saving it
    print('Now detecting and saving it')
    try:
        predictions = coco_demo.run_on_opencv_image(image)
        pil_image = Image.fromarray(predictions)

        full_image_name = hashlib.sha256(str(image_location).encode('utf-8')).hexdigest()
        print('full_image_name')
        print(full_image_name)
        full_image_name_path = folder_path + full_image_name + '.png'
        print('full_image_name_path')
        print(full_image_name_path)
        # # pil_image.save(full_image_name_path)

        # # Saving image
        images = [img_pil, pil_image]
        # images = map(Image.open, ['Test1.jpg', 'Test2.jpg', 'Test3.jpg'])
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(full_image_name_path)
        # # Saving image

        # # Now Saving to dictionary
        temp_data = {full_image_name: image_location}
        main_dict.append(temp_data)
        # # Now Saving to dictionary
        print('image saved')
        # print('dictionary upto now')
        # print(main_dict)
    except IndexError:
        print('No Image')

def fetch_vertices(vertices, image):
    """
    Gives vertices of the image
    :param vertices: actual vertices from the json
    :param image: corresponding image
    :return: clipped x, y
    """
    x = []
    y = []
    for io in range(len(vertices)):
        x.append(int(vertices[io][0]))
        y.append(int(vertices[io][1]))
    x = np.clip(x, a_min=10, a_max=image.shape[1] - 10)
    y = np.clip(y, a_min=10, a_max=image.shape[0] - 10)
    return x, y

def mask_image(json_data, folder_path, select_category, coco_demo):
    print('Start Masking Image')

    main_dict = []
    for count, key in enumerate(json_data):
        image_location = key['url']
        if not image_location:
            continue
        try:
            image = skimage.io.imread(image_location)
        except:
            continue
        if key['labels'] == None: # # Saving on another folder with tag 'no-labels'
            #save_image(image, folder_path, count, key, False, None, None, None, None)
            continue
        for i, features in enumerate(key['labels']):    # # Iterating over the labels
            category_present = category_selector(features)
            if select_category == 'all':
                category_present = features['category']
                pass
            elif category_present != select_category:
                continue
            try:
                vertices = features['poly2d'][0]['vertices']
            except IndexError:
                continue
            x, y = fetch_vertices(vertices, image)
            save_image(image, folder_path, count, key, True, x, y, category_present, i, main_dict, image_location, coco_demo)
    # # Now Saving to pickle
    dict_folder_path = folder_path + 'images.pkl'
    pickle.dump(main_dict, open(dict_folder_path, "wb"))
    # # Now Saving to pickle
    print('Masking Image Successful')

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

def main():
    """
    Masks Image and then saves it in desired location
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder_path', help="folder path to save all the images")
    parser.add_argument('-p', '--config_path', help="yml file for inference")
    parser.add_argument('-j', '--raw_json', help="raw json to bring the images")
    parser.add_argument('-c', '--category_name', help="the category name")

    args = parser.parse_args()
    make_directory(args.folder_path)
    # model = load_model(args.coco_model_path, args.detection_min_confidence)
    coco_demo = get_config(args.config_path)
    json_data = load_raw_json_data(args.raw_json)
    mask_image(json_data, args.folder_path, args.category_name, coco_demo)
    # # load_image(class_names, model, args.image_path, args.folder_path, args.category_name)

if __name__=='__main__':
    main()