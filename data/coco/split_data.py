import json
import os
import glob


def read_json(file):
    with open(file,'r') as f:
        data = json.load(file,f)
    return data


def write_json(file,data):
    with open(file,'w') as f:
        json.dump(f,data)

2
def format(annotation_file="./annotations/person_keypoints_val2017.json",image_folder="./images/1G_images/"):
    """Well, since I just download val folder of 2017 coco dataset ... I need to write a Python script to format data...
    :param annotation_file:
    :param image_folder:
    :return:
    """


