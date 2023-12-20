import os
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO

thetis_model = YOLO('yolo/yolov8l-pose.pt')

def save_video_paths_to_csv():
    """
    Save the path of videos as CSV file
    """
    base_path = "THETIS\VIDEO_RGB"

    data = []
    for motion in os.listdir(base_path):
        motion_path = os.path.join(base_path, motion)
        
        if os.path.isdir(motion_path):
            for video_name in os.listdir(motion_path):
                video_path = os.path.join(motion_path, video_name)
                data.append([motion, video_name, video_path])
                    
    df = pd.DataFrame(data, columns=["motion", "video_name", "video_path"])
    df.to_csv("csv/total_video_paths.csv", index=False)


def separate_video_paths():
    """
    Separate video paths
    """
    video_paths = 'csv/total_video_paths.csv'
    df = pd.read_csv(video_paths)

    extract_video_paths_list = []
    remaining_video_paths_list = []
    motions = ['backhand2hands', 'backhand', 'backhand_slice', 'backhand_volley', 
               'forehand_flat', 'forehand_openstands', 'forehand_slice', 'forehand_volley', 
               'flat_service', 'kick_service', 'slice_service', 'smash']
    
    for motion in motions:
        motion_video_data = df[df['motion'] == motion]

        for idx, row in motion_video_data.iterrows():
            video_name = row['video_name']
            video_path = row['video_path']

            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print(f"Error: Could not open video '{video_path}'")
                break

            pose_count = 0
            while pose_count == 0:
                ret, frame = video.read()
                if not ret:
                    break

                results = thetis_model.predict(frame)
                for result in results:
                    kpts = result.keypoints.xy
                    if len(kpts) >= 2:
                        pose_count += 1
            
            if pose_count == 0:
                extract_video_paths_list.append([motion, video_name, video_path])
            else:
                remaining_video_paths_list.append([motion, video_name, video_path])

    train_path_df = pd.DataFrame(extract_video_paths_list, columns=["motion", "video_name", "video_path"])
    train_path_df.to_csv('csv/extract_video_paths.csv', index=False)

    valid_path_df = pd.DataFrame(remaining_video_paths_list, columns=["motion", "video_name", "video_path"])
    valid_path_df.to_csv('csv/remaining_video_paths.csv', index=False)


def calculate_keypoints(image):
    """
    Calculate keypoints for a frame using a YOLO pose model
    """
    results = thetis_model.predict(image)

    for result in results:
        kpts = result.keypoints.xyn[0].cpu().numpy() # 17 x 2 배열
        keypoints_xyn = kpts.flatten() # [x1, y1, x2, y2, ...] -> 34 x 1 배열

    return keypoints_xyn


def save_keypoints(video_path, class_idx):
    """
    Process a video to calculate keypoints and appends them to a list
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return
    
    keypoints_list = []
    while True:
        ret, frame = video.read()

        if not ret:
            break

        keypoints_xyn = calculate_keypoints(frame) # YOLO 모델 적용해서 keypoints 좌표 계산

        data = np.append(keypoints_xyn, class_idx)
        keypoints_list.append(data)

    return keypoints_list


keypoints = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

def save_keypoints_to_csv(output_path, keypoints_list):
    """
    Save the collected keypoints to a CSV file
    """
    columns = [f'{keypoint}_{coord}' for keypoint in keypoints for coord in ['x', 'y']] + ['class']
    df = pd.DataFrame(keypoints_list, columns=columns)
    df.to_csv(output_path, index=False)


# 1. 모든 motion을 따로 분리
classes = {'backhand2hands': 0, 'backhand': 1, 'backhand_slice': 2, 'backhand_volley': 3, 
           'forehand_flat': 4, 'forehand_openstands': 5, 'forehand_slice': 6, 'forehand_volley': 7,  
           'flat_service': 8, 'kick_service': 9, 'slice_service': 10, 'smash': 11}

def collect_datasets():
    """
    Collect data for a specific motion from a CSV file containing video paths
    and Calculate keypoints for each frame in the videos and saves them in a CSV file.
    """
    csv_file_path = 'csv/extract_video_paths.csv'
    df = pd.read_csv(csv_file_path)

    for motion in classes.keys():
        motion_video_paths = df[df['motion'] == motion] # 특정 Motion에 대한 Video paths 추출

        class_idx = np.array([classes[motion]], dtype=np.float32) # Motion의 Label Number

        for idx, row in motion_video_paths.iterrows():
            video_path = row['video_path']

            keypoints_list = save_keypoints(video_path, class_idx) # 35 x 1 배열

            output_path = f"keypoints/{motion}/{motion}_{idx}.csv" # Keypoints를 저장할 CSV 파일
            save_keypoints_to_csv(output_path, keypoints_list) # 하나의 motion에 대한 keypoint 리스트를 csv 파일에 저장


# 2. 모든 모션을 4가지로 분리
motions = {'backhand': 0, 'forehand': 1, 'service': 2, 'smash': 3}

def collect_group_datasets():
    csv_file_path = 'csv/extract_video_paths.csv'
    df = pd.read_csv(csv_file_path)

    for motion in motions.keys():
        motion_video_paths = df[df['video_path'].str.contains(motion, case=False, regex=False)]

        class_idx = np.array([motions[motion]], dtype=np.float32) # Motion의 Label Number

        for idx, row in motion_video_paths.iterrows():
            video_path = row['video_path']

            keypoints_list = save_keypoints(video_path, class_idx) # 35 x 1 배열

            output_path = f"keypoints/{motion}/{motion}_{idx}.csv" # Keypoints를 저장할 CSV 파일
            save_keypoints_to_csv(output_path, keypoints_list) # 하나의 motion에 대한 keypoint 리스트를 csv 파일에 저장