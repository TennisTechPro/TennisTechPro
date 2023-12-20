from scipy import signal
import cv2
import numpy as np
from ultralytics import YOLO
from utils import center_of_box
from collections import defaultdict

class Tracker:
    def __init__(self):
        self.tracker = YOLO('./yolo/yolov8l.pt')
        self.player1_boxes = []
        self.player2_boxes = []
        self.track_history = defaultdict(lambda: [])
        self.persons_first_appearance = {}
        self.p1_id = 0
        self.p1_history = None
        self.p1_first_appearance_frame = 0
        self.p2_id = 0
        self.p2_history = None
        self.p2_first_appearance_frame = 0

    def track(self, frame, frame_num):
        results = self.tracker.track(frame, persist=True) # tracking
        boxes = results[0].boxes
        persons_boxes = results[0].boxes.xyxy.cpu().numpy().tolist() # 탐지한 모든 box 좌표

        if results[0].boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist() # detection 식별 번호(id)
            classes = boxes.cls.int().cpu().tolist() # class

            for box, track_id, cls in zip(persons_boxes, track_ids, classes):
                if cls == 0: # class가 person일 때만
                    if track_id not in self.persons_first_appearance.keys():
                        self.persons_first_appearance[track_id] = frame_num # 사람이 처음 등장한 프레임을 저장
                    track = self.track_history[track_id] 
                    track.append(box)

    def find_players_boxes(self):
        print('find_players...')
        dists = calculate_all_persons_dists(self.track_history) # 모든 사람의 이동 거리를 계산
        sorted_dict = sorted(dists.items(), key=lambda x: x[1], reverse=True)

        # 이동거리가 가장 큰 두 개의 아이템의 키 값을 얻음
        # 그 두 키 값이 선수들의 식별 번호(id)
        top_two_ids = [item[0] for item in sorted_dict[:2]]

        id1 = top_two_ids[0]
        id2 = top_two_ids[1]
        print(f'id1: {id1}, id2: {id2}')

        # bottom player와 top player를 구분
        self.separate_players(id1, id2)
        print(f'p1_id: {self.p1_id}, id2: {self.p2_id}')

        # 각 id에 해당하는 box 좌표를 할당
        self.p1_history = self.track_history[self.p1_id]
        self.p2_history = self.track_history[self.p2_id]

        # 선수들을 제일 처음 detection한 frame 번호
        self.p1_first_appearance_frame = self.persons_first_appearance[self.p1_id]
        self.p2_first_appearance_frame = self.persons_first_appearance[self.p2_id]

    def separate_players(self, id1, id2):
        if self.track_history[id1][self.persons_first_appearance[id1]][1] > self.track_history[id2][self.persons_first_appearance[id2]][1]:
            self.p1_id = id1
            self.p2_id = id2
        else:
            self.p1_id = id2
            self.p2_id = id1

    def mark_boxes(self, frame, frame_num):
        # 현재 프레임보다 같거나 이전에 선수를 처음 탐지했을 때 box를 그림
        if self.p1_first_appearance_frame <= frame_num and self.p1_first_appearance_frame + len(self.p1_history) > frame_num:
            p1_boxes = self.p1_history[frame_num - self.p1_first_appearance_frame]
            frame = cv2.rectangle(frame, (int(p1_boxes[0]), int(p1_boxes[1])), (int(p1_boxes[2]), int(p1_boxes[3])), [255, 0, 255], 2)
        else:
            p1_boxes = None
        self.player1_boxes.append(p1_boxes)

        if self.p2_first_appearance_frame <= frame_num and self.p2_first_appearance_frame + len(self.p2_history) > frame_num:
            p2_boxes = self.p2_history[frame_num - self.p2_first_appearance_frame]
            frame = cv2.rectangle(frame, (int(p2_boxes[0]), int(p2_boxes[1])), (int(p2_boxes[2]), int(p2_boxes[3])), [255, 255, 0], 2)
        else:
            p2_boxes = None
        self.player2_boxes.append(p2_boxes)

        return frame, p1_boxes, p2_boxes
    
    def calculate_feet_positions(self, court_detector):
        """
        Calculate the feet position of both players using the inverse transformation of the court and the boxes of both players
        """
        inv_mats = court_detector.game_warp_matrix
        positions_1 = []
        positions_2 = []

        mask_1 = []
        mask_2 = []

        # Bottom player feet locations
        for i, box in enumerate(self.player1_boxes):
            if box is not None:
                feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2), box[3]]).reshape((1, 1, 2))
                feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
                positions_1.append(feet_court_pos)
                mask_1.append(True)
            elif len(positions_1) > 0:
                positions_1.append(positions_1[-1])
                mask_1.append(False)
            else:
                positions_1.append(np.array([0, 0]))
                mask_1.append(False)

        # Top player feet locations
        for i, box in enumerate(self.player2_boxes):
            if box is not None:
                feet_pos = np.array([(box[0] + (box[2] - box[0]) / 2), box[3]]).reshape((1, 1, 2))
                feet_court_pos = cv2.perspectiveTransform(feet_pos, inv_mats[i]).reshape(-1)
                positions_2.append(feet_court_pos)
                mask_2.append(True)
            elif len(positions_2) > 0:
                positions_2.append(positions_2[-1])
                mask_2.append(False)
            else:
                positions_2.append(np.array([0, 0]))
                mask_2.append(False)
        
        # Smooth both feet locations
        positions_1 = np.array(positions_1)
        smoothed_1 = np.zeros_like(positions_1)
        smoothed_1[:, 0] = signal.savgol_filter(positions_1[:, 0], 7, 2)
        smoothed_1[:, 1] = signal.savgol_filter(positions_1[:, 1], 7, 2)
        smoothed_1[not mask_1, :] = [None, None]

        positions_2 = np.array(positions_2)
        smoothed_2 = np.zeros_like(positions_2)
        smoothed_2[:, 0] = signal.savgol_filter(positions_2[:, 0], 7, 2)
        smoothed_2[:, 1] = signal.savgol_filter(positions_2[:, 1], 7, 2)
        smoothed_2[not mask_2, :] = [None, None] # input video 2에서는 top 선수 탐지 X -> TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'

        return smoothed_1, smoothed_2
    
    def print_counts(self):
        print('Player 1 Object Detection Count: ', len(self.player1_boxes))
        print('Player 2 Object Detection Count: ', len(self.player2_boxes))


def calculate_all_persons_dists(persons_boxes):
    persons_dists = {}
    for id, person_boxes in persons_boxes.items():
        person_boxes = [box for box in person_boxes if box[0] is not None]
        dist = boxes_dist(person_boxes)
        persons_dists[id] = dist
    return persons_dists

def boxes_dist(boxes):
    total_dist = 0
    for box1, box2 in zip(boxes, boxes[1:]):
        box1_center = np.array(center_of_box(box1))
        box2_center = np.array(center_of_box(box2))
        dist = np.linalg.norm(box2_center - box1_center)
        total_dist += dist
    return total_dist