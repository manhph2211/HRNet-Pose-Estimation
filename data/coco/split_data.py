import json
import os
from tqdm import tqdm
from shutil import copyfile


def read_json(file):
    with open(file,'r') as f:
        data = json.load(f)
    return data


def write_json(file,data):
    with open(file,'w') as f:
        json.dump(data,f)


def format(annotation_file="./annotations/in_images.json",image_folder="./images/in_images/",person_detection_results_file='./person_detection_results/in_detection_results.json',val_size = 0.2):
    """Well, since I just download val folder of 2017 coco dataset ... I need to write a Python script to format data...
    :param annotation_file:
    :param image_folder:
    :param person_detection_results_file:
    :param val_size:
    :return:
    """
    full_data = read_json(annotation_file)
    info = full_data['info']
    licenses = full_data['licenses']
    categories = full_data['categories']
    train_dict = {'info':info,'licenses':licenses,'categories':categories,'images':[],'annotations':[]}
    val_dict = {'info':info,'licenses':licenses,'categories':categories,'images':[],'annotations':[]}
    person_detection_results = read_json(person_detection_results_file)
    new_person_detection_results = []
    if not os.path.isdir('./images/train2017'):
        os.mkdir('./images/train2017')
    if not os.path.isdir('./images/val2017'):
        os.mkdir('./images/val2017')

    for i,image in tqdm(enumerate(full_data['images'])):
        id_image = image['id']
        file_name = image['file_name']
        for anno in full_data['annotations']:
            if anno['image_id'] == id_image:
                if i <= len(full_data['images'])*val_size:
                    val_dict['images'].append(image)
                    val_dict['annotations'].append(anno)
                    copyfile(os.path.join(image_folder,file_name),os.path.join('./images/val2017',file_name))
                    for person_detection_result in person_detection_results:
                        if person_detection_result['image_id'] == id_image:
                            new_person_detection_results.append(person_detection_result)
                else:
                    train_dict['images'].append(image)
                    train_dict['annotations'].append(anno)
                    copyfile(os.path.join(image_folder,file_name),os.path.join('./images/train2017',file_name))
                break # NOT SURE
    write_json('./annotations/person_keypoints_train2017.json',train_dict)
    write_json('./annotations/person_keypoints_val2017.json', val_dict)
    write_json('./person_detection_results/COCO_val2017_detections_AP_H_56_person.json', new_person_detection_results)


if __name__ =="__main__":
    print("CONVERTING...")
    format()


