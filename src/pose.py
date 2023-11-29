import os
import cv2
import torch
import numpy as np
import pandas as pd

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


class PoseExtractor:
    def __init__(self, person_num=2, box=False, dtype=torch.FloatTensor):
        self.pose_model = YOLO('yolo/yolov8n-pose.pt')
        self.dtype = dtype
        self.person_num = person_num  # 단식: 2명, 복식: 4명
        self.box = box # box 표시할건지
        self.PERSON_LABEL = None
        self.SCORE_MIN = 0.90
        self.keypoint_threshold = 2
        self.p1_keypoints = []
        self.p2_keypoints = []
        self.line_width = 2
        self.margin = 15
        self.player_1_count = 0
        self.player_2_count = 0
        self.line_connection = [(7, 9), (7, 5), (10, 8), (8, 6), (6, 5), (15, 13),
                                (13, 11), (11, 12), (12, 14), (14, 16), (5, 11), (12, 6)] # Court Line
        self.COCO_PERSON_KEYPOINT_NAMES = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ] # 'left_eye', 'right_eye', 'left_ear', 'right_ear' 제거
        self.AIHUB_PERSON_KEYPOINT_NAMES = [
            'head', 'neck', 'chst', 'bely', 'sdlf', 'sdrt', 'eblf', 'ebrt', 'wtlf', 'wtrt', 'hplf', 'hprt', 'knlf', 'knrt', 'aklf', 'akrt'
            ]
        self.player = ['target']


    def extract_pose(self, image, p1_boxes, p2_boxes):
        """
        Extract poses from the given image for both players
        """
        result = self._extract_player1_pose(image, p1_boxes)
        result = self._extract_player2_pose(result, p2_boxes)

        return result


    def _extract_player1_pose(self, image, p1_boxes):
        """
        Extract the pose of player 1(bottom) from the given image
        """
        height, width = image.shape[:2]

        if p1_boxes is not None:
            x1, y1, x2, y2 = p1_boxes # p1_boxes[-1]
            xt, yt, xb, yb = int(x1), int(y1), int(x2), int(y2)
            patch = image[max(yt - self.margin, 0):min(yb + self.margin, height), max(xt - self.margin, 0):min(xb + self.margin, width)].copy() # copy 안하면 오류남^^

            p1_patch, kpts = self._annotate_pose_on_patch(patch)
            image[max(yt - self.margin, 0):min(yb + self.margin, height), max(xt - self.margin, 0):min(xb + self.margin, width)] = p1_patch
            
            if len(kpts) != 0:
                self.p1_keypoints.append(kpts)
                self.player_1_count += 1
            else:
                self.p1_keypoints.append(np.zeros(26))
        else:
            self.p1_keypoints.append(np.zeros(26))

        return image


    def _extract_player2_pose(self, image, p2_boxes):
        """
        Extract the pose of player 2(top) from the given image
        """
        height, width = image.shape[:2]

        if p2_boxes is not None:
            x1, y1, x2, y2 = p2_boxes # p2_boxes[-1]
            xt, yt, xb, yb = int(x1), int(y1), int(x2), int(y2)
            patch = image[max(yt - self.margin, 0):min(yb + self.margin, height), max(xt - self.margin, 0):min(xb + self.margin, width)].copy() # copy 안하면 오류남^^
            
            p2_patch, kpts = self._annotate_pose_on_patch(patch)
            image[max(yt - self.margin, 0):min(yb + self.margin, height), max(xt - self.margin, 0):min(xb + self.margin, width)] = p2_patch
            
            if len(kpts) != 0:
                self.p2_keypoints.append(kpts)
                self.player_2_count += 1
            else:
                self.p2_keypoints.append(np.zeros(26))
        else:
            self.p2_keypoints.append(np.zeros(26))
            
        return image
    

    def _annotate_pose_on_patch(self, patch):
        """
        Annotate the pose on a patch of the image
        """
        results = self.pose_model.predict(patch, boxes=False)
        kpts = results[0].keypoints.xyn[0].cpu().numpy().flatten()

        # ear와 eye에 관련된 값 삭제
        kpts = np.concatenate([kpts[:2], kpts[10:]])

        player_patch = results[0].plot(kpt_radius=3, line_width=self.line_width, boxes=False)

        return player_patch, kpts
    
    
    def print_counts(self):
        print('Player 1 Pose Estimation Count: ', self.player_1_count)
        print('Player 2 Pose Estimation Count: ', self.player_2_count)


    def save_to_csv(self):
        columns = [f'{keypoint}_{coord}' for keypoint in self.COCO_PERSON_KEYPOINT_NAMES for coord in ['x', 'y']]

        # player1의 keypoint 좌표를 모두 저장
        p1_df = pd.DataFrame(self.p1_keypoints, columns=columns)
        p1_df.to_csv('output/keypoints/p1_keypoints.csv', index=False)

        # player2의 keypoint 좌표를 모두 저장
        p2_df = pd.DataFrame(self.p2_keypoints, columns=columns)
        p2_df.to_csv('output/keypoints/p2_keypoints.csv', index=False)

        return p1_df, p2_df