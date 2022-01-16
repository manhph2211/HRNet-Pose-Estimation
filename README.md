# Pose-Estimation
This repo is a sub-task of the smart hospital bed project which is about implementing the task of pose estimation :smile: Many thanks to the main authors of [hrnet-human-pose-estimation project](https://github.com/HRNet/HRNet-Human-Pose-Estimation) - the project I'd mainly based on.
## Setup
### Data preparation
- I used coco dataset(2017) which can be downloaded [here](https://cocodataset.org/#download), and the format is just like this:
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

### Dependencies
- Here are required packages:
```angular2html
torch
torchvision
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
## Usage 
```angular2html
pip install -r requirements.txt
cd src
python3 main.py # update later
```
## Citation
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