import os
import glob
import json
import yaml

import argparse
import cv2
import torch
import numpy as np
import pandas as pd

def convert_bbox_to_yolo_format(size, bbox):
    """
    Convert bounding box coordinates from bbox format to YOLO format.

    Args:
        size (tuple): Tuple containing the width and height of the image.
        bbox (list): List containing the coordinates of the bounding box in the format [x, y, w, h].

    Returns:
        list: List containing the coordinates of the bounding box in YOLO format [center_x, center_y, w, h].
    """
    dw = 1 / size[0]
    dh = 1 / size[1]

    w = bbox[2]
    h = bbox[3]
    x = bbox[0] + (w / 2) # center_x
    y = bbox[1] + (h / 2) # center_y

    x = round(x * dw, 6)
    y = round(y * dh, 6)
    w = round(w * dw, 6)
    h = round(h * dh, 6)

    return [x, y, w, h]


def make_yolo_dataset(json_dir, txt_dir, img_size):
    """
    Convert JSON annotations to YOLO format and save them as text files.

    Args:
        json_dir (str): Directory containing the JSON annotation files.
        txt_dir (str): Directory to save the YOLO format text files.
        img_size (int): Size of the images.

    Returns:
        None
    """
    for json_name in os.listdir(json_dir):
        if json_name.endswith('json'):
            json_path = os.path.join(json_dir, json_name)
            data = json.load(open(json_path, encoding='utf-8'))

            dataInfo = data['learningDataInfo']

            # img_name: [테니스, 날짜, (2차 경로), (3차 경로), 카메라번호, 원시데이터일련번호, 동작분석, 시퀀스일련번호, 동작분석클래스, 동작분석일련번호, 이미지일련번호]
            img_name = json_name.replace('.json', '')
            class_name = img_name.split('_')[-3] # Motion Estimation에 사용

            # annotation 정보
            annotations = dataInfo['annotations']

            # Court and Ball Detection에 사용
            court_polygon = annotations[0]['polygon']
            court_type = int(data['rawDataInfo']['courtNumber'].replace('코트', '')) # 코트 Number
            net_polygon = annotations[1]['polygon']

            bbox_list = []

            # Player1: bottom player(아래쪽에 위치한 선수)
            player1_bbox = annotations[2]['bbox']
            bbox_list.append([0] + convert_bbox_to_yolo_format(img_size, player1_bbox)) # class 0: bottom player

            # Player2: top player(위쪽에 위치한 선수)
            player2_bbox = annotations[3]['bbox']
            bbox_list.append([1] + convert_bbox_to_yolo_format(img_size, player2_bbox)) # class 1: top player

            # Save txt file
            with open(txt_dir + img_name + '.txt', 'w') as f:
                for bbox in bbox_list:
                    data = ' '.join(map(str, bbox))
                    f.write(data + '\n')


def remove_txt_files(txt_dir):
    """
    Remove text files from the specified directory
    """
    bbox_txt_list = glob.glob(txt_dir + '*.txt')
    for i in bbox_txt_list:
        os.remove(i)


def make_yolo_yml():
    """
    Create YOLO configuration file.
    """
    with open("./yolo/object_classes.txt", "r") as reader:
        lines = reader.readlines()
        classes = [line.strip().split(",")[1] for line in lines]

    path = os.getcwd()

    yaml_data = {
                  "names": classes,
                  "nc": len(classes),
                  "path": path,
                  "train": "./datasets/train",
                  "val": "./datasets/valid",
                }

    with open("./yolo/detection_config.yaml", "w") as writer:
        yaml.dump(yaml_data, writer)


def make_pose_csv():
    rootdir = 'datasets/THETIS/VIDEO_RGB/'
    data = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            data.append([os.path.split(subdir)[-1], file])
            print([os.path.split(subdir)[-1], file])
    df = pd.DataFrame(data, columns=['folder', 'name'])
    outfile_path = os.path.join(rootdir, 'THETIS_data.csv')
    df.to_csv(outfile_path, index=False)


# 이미지의 센터를 왜 자르는지 아직까진 모르겠음
def crop_center(image):
    # crop the center of an image and matching the height with the width of the image
    shape = image.shape[:-1]
    max_size_index = np.argmax(shape)
    diff1 = abs((shape[0] - shape[1]) // 2)
    diff2 = shape[max_size_index] - shape[1 - max_size_index] - diff1
    return image[:, diff1: -diff2] if max_size_index == 1 else image[diff1: -diff2, :]


# CPU 사용할건지 GPU 사용할건지
def get_dtype():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    if dev == 'cuda':
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    print(f'Using device {device}')
    return dtype


# OpenCV 버전에 따라 영상의 properties를 추출함
def get_video_properties(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # get videos properties
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        length = int(video.get(cv2.cv.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_stickman_line_connection():
    # stick man line connection with keypoints indices for R-CNN
    # R-CNN 말고 YOLO에도 활용 가능?
    line_connection = [
        (7, 9), (7, 5), (10, 8), (8, 6), (6, 5), (15, 13), (13, 11), (11, 12), (12, 14), (14, 16), (5, 11), (12, 6)
    ]
    return line_connection


def center_of_box(box):
    if box[0] is None:
        return None, None
    height = box[3] - box[1]
    width = box[2] - box[0]
    return box[0] + width / 2, box[1] + height / 2