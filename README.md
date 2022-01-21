Pose-Estimation
=====

- This repo is a sub-task of the smart hospital bed project which is about implementing the task of pose estimation :smile: Many thanks to the main authors of [hrnet-human-pose-estimation project](https://github.com/HRNet/HRNet-Human-Pose-Estimation) - the project I'd mainly based on.

- [Hrnet](https://arxiv.org/pdf/1902.09212.pdf) wants to maintain reliable high-resolution representations of images through the entire training process, besides fusioning different high-to-low sub-networks, which is special contribution of the paper. 
![image](https://user-images.githubusercontent.com/61444616/150523678-45585974-a960-474f-8166-ffb51009f38b.png)

- There are two stages here. First, you need to use a object detection algorithm to get bounding box of person in image. The each predicted person region will be cut from original image and then fed into hrnet-pose-estimation for our main purpose!

- Here `coco` format was used. When you are labeling, you might wanna label box of person and keypoints of that person!

# Setup
## Data preparation
- Coco dataset(2017) which can be downloaded [here](https://cocodataset.org/#download), and the format is just like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```
- Note that `person_detection_results` contains json results of particular perseon detector. It's not really necessary when dealing with COCO datasets because they provide it already, but when it comes to your custom dataset, you should consider to train and get predicted bboxes for person in images, and the save it in `person_detection_results` as json format before coming to pose estimation stage!

## Dependencies
- Here are required packages:
```angular2html
torch
torchvision
opencv-python
numpy
json_tricks
yacs>=0.1.5
Cython
```
- You also need to install [cocoapi](https://github.com/cocodataset/cocoapi):
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python3 setup.py install --user
cd ..
```
- For object detector, I would like to use `yolov5`, you can go to the [official implementation](https://github.com/ultralytics/yolov5) for installation. Or you can follow:
```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install

``` 

# Usage 

- Following these:
```angular2html
pip install -r requirements.txt
cd src
python3 main.py # update later
```

# Citation
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```
