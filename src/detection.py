import torch
from scipy import signal
import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class DetectionModel:
    def __init__(self, match_type=2, dtype=torch.FloatTensor):
        self.detection_model = YOLO('yolo/clip.pt')
        self.match_type = match_type # 단식 2명, 복식 4명
        self.dtype = dtype
        self.BOTTOM_PLAYER_1 = 0
        self.BOTTOM_PLAYER_2 = 2
        self.TOP_PLAYER_1 = 1
        self.TOP_PLAYER_2 = 3
        self.BOTTOM_MIN_CONF = 0.55 # Bottom 선수 Detection 시 최소 confidence 값
        self.TOP_MIN_CONF = 0.32 # Top 선수 Detection 시 최소 confidence 값
        self.line_thickness = 2
        self.player_1_boxes = []
        self.player_2_boxes = []
        self.player_1_count = 0
        self.player_2_count = 0
        self.v_height = 0
        self.v_width = 0
        self.cls = []


    # def detect(self, image, person_min_score=None):
    #     """
    #     Use YOLO Model to detect players in the one frame
    #     """
    #     results = self.detection_model.predict(image)

    #     for result in results:
    #         annotator = Annotator(image, line_width=self.line_thickness, font_size=None)

    #         boxes = result.boxes

    #         # Model 성능 보고 더 간단하게 만들 것
    #         # 현재 모델의 Detection in Input Video1 : 2 bottom (대다수), 1 bottom 1 top (극소수), 2 bottom 1 top (극극소수)로 탐지
    #         # No detection은 알 바 아님
    #         if len(boxes.xyxy) == 3:
    #             box_1 = boxes.xyxy[0].cpu().numpy()
    #             cls_1 = boxes.cls[0].cpu().numpy()
    #             conf_1 = boxes.conf[0].cpu().numpy()
    #             cls = boxes.cls[1].cpu().numpy()
    #             box_2 = boxes.xyxy[2].cpu().numpy()
    #             cls_2 = boxes.cls[2].cpu().numpy()
    #             conf_2 = boxes.conf[2].cpu().numpy()
    #             self.player_1_boxes.append(box_1)
    #             self.player_2_boxes.append(box_2)
    #             annotator.box_label(box_1, label=False, color=(255, 0, 0))
    #             annotator.box_label(box_2, label=False, color=(255, 0, 0))
    #             self.cls.append([cls_1, cls, cls_2])

    #         elif len(boxes.xyxy) == 2:
    #             box_1 = boxes.xyxy[0].cpu().numpy()
    #             cls_1 = boxes.cls[0].cpu().numpy()
    #             conf_1 = boxes.conf[0].cpu().numpy()
    #             box_2 = boxes.xyxy[1].cpu().numpy()
    #             cls_2 = boxes.cls[1].cpu().numpy()
    #             conf_2 = boxes.conf[1].cpu().numpy()

    #             self.player_1_boxes.append(box_1)
    #             self.player_2_boxes.append(box_2)
    #             self.player_1_count += 1
    #             self.player_2_count += 1
    #             annotator.box_label(box_1, label=False, color=(255, 0, 0))
    #             annotator.box_label(box_2, label=False, color=(255, 0, 0))

    #         elif len(boxes.xyxy) == 1:
    #             box = boxes.xyxy[0].cpu().numpy()
    #             cls = boxes.cls[0].cpu().numpy()
    #             conf = boxes.conf[0].cpu().numpy()
    #             annotator.box_label(box, label=False, color=(255, 0, 0))

    #             if cls == 0:
    #                 self.player_1_boxes.append(box)
    #                 self.player_1_count += 1
    #                 self.player_2_boxes.append(None)
    #             elif cls == 1:
    #                 self.player_1_boxes.append(None)
    #                 self.player_2_boxes.append(box)
    #                 self.player_2_count += 1

    #         else:
    #             self.player_1_boxes.append(None)
    #             self.player_2_boxes.append(None)
                
    #     frame = annotator.result()
    #     return frame
    
    def detect(self, image):
        persons_boxes = []

        results = self.detection_model.predict(image)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls.cpu().numpy())
                if cls == 0:
                    persons_boxes.append(box.xyxy[0].cpu().numpy())

        return persons_boxes


    def detect_bottom_player(self, frame, court_detector):
        # 일단 bottom에 있는 선수만 처음 프레임에서 잘 찾는다면 모델 성능은 크게 상관없음
        self.v_height, self.v_width = frame[:2]

        if len(self.player_1_boxes) == 0:
            court_type = 1
            white_ref = court_detector.court_reference.get_court_mask(court_type)
            white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix[-1], frame.shape[1::-1])

            image_court = frame.copy()
            image_court[white_mask == 0, :] = (0, 0, 0)

            persons_boxes = self.detect(image_court) # Detect한 모든 Box 좌표

            if len(persons_boxes) > 0:
                bottom_box = None
                smallest_dist = np.inf

                for box in persons_boxes:
                    y2 = box[3] # Box의 우측 하단 y 좌표 활용
                    dist = self.v_height - y2 # frame의 height와 우측 하단 y 좌표의 차이 계산
                    if self.v_height - y2 < smallest_dist: # 차이가 가장 작은 box를 bottom 선수의 좌표로 선택
                        smallest_dist = dist
                        bottom_box = box # 경기의 첫 시작은 무조건 테니스 코트 아래에서 시작하기 때문

                self.player_1_boxes.append(bottom_box)
        else:
            # 이전의 player의 위치를 통해 새로운 ROI 적용
            x1, y1, x2, y2 = self.player_1_boxes[-1] 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            margin = 250
            xt, yt, xb, yb = max(x1 - margin, 0), max(y1 - margin, 0), min(x2 + margin, self.v_width), min(y2 + margin, self.v_height)
            roi_corner = [xt, yt, xb, yb]
            roi_img = frame[yt : yb, xt : xb, :]

            persons_boxes = self.detect(roi_img)

            if len(persons_boxes) > 0:
                c1 = center_of_box(self.player_1_boxes[-1])
                closest_box = None
                smallest_dist = np.inf

                for box in persons_boxes:
                    c2 = center_of_box(box)
                    origin_box = [xt + box[0], yt + box[1], xt + box[2], yt + box[3]]
                    dist = np.linalg.norm(np.array(c1) - np.array(c2))
                    if dist < smallest_dist:
                        smallest_dist = dist
                        closest_box = origin_box
                self.player_1_boxes.append(closest_box)

        # results = self.detection_model.predict(image_court)

        # for result in results:
        #     annotator = Annotator(frame, line_width=self.line_thickness, font_size=None)
        #     boxes = result.boxes

        #     if len(boxes) != 0:
        #         box = boxes.xyxy[0].cpu().numpy()
        #         cls = boxes.cls[0].cpu().numpy()
        #         conf = boxes.conf[0].cpu().numpy()
        #         self.player_1_boxes.append(box)
        #         annotator.box_label(box, label=False, color=(255, 0, 0))
        #         self.player_1_count += 1
        #     else:
        #         self.player_1_boxes.append(None)
                
        # frame = annotator.result()
        # return frame
    

    def detect_top_player(self, frame, court_detector):
        court_type = 2
        white_ref = court_detector.court_reference.get_court_mask(court_type)
        white_mask = cv2.warpPerspective(white_ref, court_detector.court_warp_matrix[-1], frame.shape[1::-1])
        white_mask = cv2.dilate(white_mask, np.ones((100, 1)), anchor=(0, 0))

        image_court = frame.copy()
        image_court[white_mask == 0, :] = (0, 0, 0)

        results = self.detection_model.predict(image_court)

        for result in results:
            annotator = Annotator(frame, line_width=self.line_thickness, font_size=None)
            boxes = result.boxes

            if len(boxes) != 0:
                box = boxes.xyxy[0].cpu().numpy()
                cls = boxes.cls[0].cpu().numpy()
                conf = boxes.conf[0].cpu().numpy()
                self.player_2_boxes.append(box)
                annotator.box_label(box, label=False, color=(255, 0, 0))
                self.player_2_count += 1
            else:
                self.player_2_boxes.append(None)

        frame = annotator.result()
        return frame
    
    def preprocessing_box_coordinates(self):
        positions_1 = []
        positions_2 = []

        mask_1 = []
        mask_2 = []

        for i, box in enumerate(self.player_1_boxes):
            if box is not None:
                positions_1.append(box)
                mask_1.append(True)
            elif len(positions_1) > 0:
                positions_1.append(positions_1[-1])
                mask_1.append(False)
            else:
                positions_1.append(np.array([0, 0]))
                mask_1.append(False)
            
        for i, box in enumerate(self.player_2_boxes):
            if box is not None:
                positions_2.append(box)
                mask_2.append(True)
            elif len(positions_2) > 0:
                positions_2.append(positions_2[-1])
                mask_2.append(False)
            else:
                positions_2.append(np.array([0, 0]))
                mask_2.append(False)

        # Smooth both locations
        positions_1 = np.array(positions_1)
        smoothed_1 = np.zeros_like(positions_1)
        smoothed_1[:, 0] = signal.savgol_filter(positions_1[:, 0], 7, 2)
        smoothed_1[:, 1] = signal.savgol_filter(positions_1[:, 1], 7, 2)
        smoothed_1[not mask_1, :] = [None, None]
        self.player_1_boxes = smoothed_1

        positions_2 = np.array(positions_2)
        smoothed_2 = np.zeros_like(positions_2)
        smoothed_2[:, 0] = signal.savgol_filter(positions_2[:, 0], 7, 2)
        smoothed_2[:, 1] = signal.savgol_filter(positions_2[:, 1], 7, 2)
        smoothed_2[not mask_2, :] = [None, None]
        self.player_2_boxes = smoothed_2

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
        for i, box in enumerate(self.player_1_boxes):
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
        for i, box in enumerate(self.player_2_boxes):
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

    def get_boxes(self):
        return self.player_1_boxes, self.player_2_boxes
    
    def print_counts(self):
        print('Player 1 Object Detection Count: ', self.player_1_count)
        print('Player 2 Object Detection Count: ', self.player_2_count)



def center_of_box(box):
    """
    Calculate the center of a box
    """
    if box[0] is None:
        return None, None
    height = box[3] - box[1]
    width = box[2] - box[0]
    return box[0] + width / 2, box[1] + height / 2