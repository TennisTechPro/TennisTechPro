from sympy import Line
from itertools import combinations, product
from court_reference import CourtReference
import matplotlib.pyplot as plt

import numpy as np
import cv2

import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from postprocess import postprocess, refine_kps, line_intersection_post
from scipy.spatial import distance

class CourtDetector:
    # Frame에서 테니스 코트를 추출하고 tracking 하는 클래스
    def __init__(self, verbose=0, out_channels=15):
        self.verbose = verbose # 출력문 flag -> 0이면 x
        self.court_reference = CourtReference() # Reference Court
        
        self.frame = None # 영상의 한 프레임을 읽어서 저장
        self.v_width = 0 # Frame의 가로
        self.v_height = 0 # Frame의 세로
        self.model_output_width = 640
        self.model_output_height = 360
        
        # pretrained model 불러오기
        self.device = torch.device('cpu')
        self.detector = BallTrackerNet(out_channels=out_channels)
        self.detector = self.detector.to(self.device)
        self.detector.load_state_dict(torch.load('models/model_tennis_court_det.pt', map_location=self.device))
        self.detector.eval()

        self.court_warp_matrix = [] # transformation matrix
        self.game_warp_matrix = [] # transformation inverse matrix
        self.best_conf = None
        '''
        self.baseline_top = None
        self.baseline_bottom = None
        self.net = None
        self.left_court_line = None
        self.right_court_line = None
        self.left_inner_line = None
        self.right_inner_line = None
        self.middle_line = None
        self.top_inner_line = None
        self.bottom_inner_line = None
        '''
        self.frame_points = None
        self.dist = 5

    def detect(self, frame):
        # 프레임(영상)에서 테니스 코트를 추출하는 함수

        # 각 변수 초기화
        self.frame = frame
        self.v_height, self.v_width = frame.shape[:2]

        # 이미지 전처리
        img = self._preprocess_img(frame)

        # 테니스 코트 14개 keypoint 예측
        out = self.detector(img.float().to(self.device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        # 이미지 후처리
        points = self._postprocess_img(self.frame, pred)

        # 이미지에서 추출한 테니스 코트와
        # Reference court 사이의 transformation을 계산
        # homography matrix 생성 후 변환
        court_warp_matrix, game_warp_matrix = self._find_homography(points)
        
        self.court_warp_matrix.append(court_warp_matrix)
        self.game_warp_matrix.append(game_warp_matrix)

        # 프레임에서 테니스 코트의 중요한 선들의 위치를 저장
        lines = self.find_lines_location()

        return lines
  

    def _preprocess_img(self, frame):
        # 이미지 전처리
        img = cv2.resize(frame, (self.model_output_width, self.model_output_height))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)
        
        return inp
  

    def _postprocess_img(self, frame, pred):
        # 이미지 후처리
        points = []
        for kps_num in range(14):
            # 히트맵 기반 이미지 후처리
            heatmap = (pred[kps_num]*255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            # keypoint 보정 함수 사용 -> keypoint 더 정확하게 조정
            if kps_num not in [8, 9, 12] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(frame, int(y_pred), int(x_pred))
            points.append((x_pred, y_pred))
        
        return points

    
    def _find_homography(self, points):
        # Reference court랑 이미지의 테니스 코트 사이의 transformation 계산
        court_ref = self.court_reference
        refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

        court_conf_ind = {}
        for i in range(len(court_ref.court_conf)):
            conf = court_ref.court_conf[i+1]
            inds = []
            for j in range(4):
                inds.append(court_ref.key_points.index(conf[j]))
            court_conf_ind[i+1] = inds
        
        max_mat, max_inv_mat = self._get_trans_matrix(court_ref, court_conf_ind, points, refer_kps)
        if max_mat is not None:
            points = cv2.perspectiveTransform(refer_kps, max_mat)
            points = [np.squeeze(x) for x in points]

        return max_mat, max_inv_mat 
    
    def _get_trans_matrix(self, court_ref, court_conf_ind, points, refer_kps):
        dist_max = np.Inf
        matrix_trans = None
        inv_matrix_trans = None

        for conf_ind in range(1, 13):
            conf = court_ref.court_conf[conf_ind]

            inds = court_conf_ind[conf_ind]
            inters = [points[inds[0]], points[inds[1]], points[inds[2]], points[inds[3]]]
            if not any([None in x for x in inters]):
                matrix, _ = cv2.findHomography(np.float32(conf), np.float32(inters), method=0)
                inv_matrix = cv2.invert(matrix)[1]
                trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
                dists = []
                for i in range(12):
                    if i not in inds and points[i][0] is not None:
                        # 두 지점 간의 유클리디안 거리 계산 -> 변환의 정확도 확인
                        dists.append(distance.euclidean(np.array(points[i]).flatten(), np.array(trans_kps[i]).flatten()))
                dist_median = np.mean(dists)
                if dist_median < dist_max:
                    matrix_trans = matrix
                    inv_matrix_trans = inv_matrix
                    dist_max = dist_median
                    self.best_conf = conf_ind
        return matrix_trans, inv_matrix_trans

    def find_lines_location(self):
        # 테니스 코트에서 주요 직선들의 위치를 찾음

        p = np.array(self.court_reference.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
        lines = cv2.perspectiveTransform(p, self.court_warp_matrix[-1]).reshape(-1)

        # lines의 좌표를 court의 각 부분에 할당

        self.baseline_top = lines[:4]
        self.baseline_bottom = lines[4:8]
        self.net = lines[8:12]
        self.left_court_line = lines[12:16]
        self.right_court_line = lines[16:20]
        self.left_inner_line = lines[20:24]
        self.right_inner_line = lines[24:28]
        self.middle_line = lines[28:32]
        self.top_inner_line = lines[32:36]
        self.bottom_inner_line = lines[36:40]

        return lines
    
    def track_court(self, frame):
        # 프레임에서 감지된 테니스 코트의 위치를 추적

        copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.frame_points is None:
            conf_points = np.array(self.court_reference.court_conf[self.best_conf], dtype=np.float32).reshape((-1, 1, 2))
            self.frame_points = cv2.perspectiveTransform(conf_points, self.court_warp_matrix[-1]).squeeze().round()
            
        # Lines of configuration on frames
        line1 = self.frame_points[:2]
        line2 = self.frame_points[2:4]
        line3 = self.frame_points[[0, 2]]
        line4 = self.frame_points[[1, 3]]
        lines = [line1, line2, line3, line4]
        new_lines = []

        for line in lines:
            # Get 100 samples of each line in the frame
            points_on_line = np.linspace(line[0], line[1], 102)[1:-1]  # 100 samples on the line
            p1 = None
            p2 = None
            if line[0][0] > self.v_width or line[0][0] < 0 or line[0][1] > self.v_height or line[0][1] < 0:
                for p in points_on_line:
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p1 = p
                        break
            if line[1][0] > self.v_width or line[1][0] < 0 or line[1][1] > self.v_height or line[1][1] < 0:
                for p in reversed(points_on_line):
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p2 = p
                        break
            # if one of the ends of the line is out of the frame get only the points inside the frame
            if p1 is not None or p2 is not None:
                print('points outside screen')
                points_on_line = np.linspace(p1 if p1 is not None else line[0], p2 if p2 is not None else line[1], 102)[1:-1]

            new_points = []

            # Find max intensity pixel near each point
            for p in points_on_line:
                p = (int(round(p[0])), int(round(p[1])))
                top_y, top_x = max(p[1] - self.dist, 0), max(p[0] - self.dist, 0)
                bottom_y, bottom_x = min(p[1] + self.dist, self.v_height), min(p[0] + self.dist, self.v_width)
                patch = gray[top_y: bottom_y, top_x: bottom_x]
                y, x = np.unravel_index(np.argmax(patch), patch.shape)
                if patch[y, x] > 150:
                    new_p = (x + top_x + 1, y + top_y + 1)
                    new_points.append(new_p)
                    cv2.circle(copy, p, 1, (255, 0, 0), 1)
                    cv2.circle(copy, new_p, 1, (0, 0, 255), 1)
            new_points = np.array(new_points, dtype=np.float32).reshape((-1, 1, 2))

            # find line fitting the new points
            [vx, vy, x, y] = cv2.fitLine(new_points, cv2.DIST_L2, 0, 0.01, 0.01)
            new_lines.append(((int(x - vx * self.v_width), int(y - vy * self.v_width)),
                              (int(x + vx * self.v_width), int(y + vy * self.v_width))))

            # if less than 50 points were found detect court from the start instead of tracking
            if len(new_points) < 50:
                print('Camera ...', end=' ')
                if self.dist > 20:
                    print('Has benn Moved')
                    # cv2.imshow('court', copy)
                    # if cv2.waitKey(0) & 0xff == 27:
                    #     cv2.destroyAllWindows()
                    return self.detect(frame)
                else:
                    print('Court tracking failed, adding 5 pixels to dist')
                    self.dist += 5
                    # self.track_court(frame)
                    # return
                    return self.track_court(frame)
                
        # Find transformation from new lines
        i1 = line_intersection(new_lines[0], new_lines[2])
        i2 = line_intersection(new_lines[0], new_lines[3])
        i3 = line_intersection(new_lines[1], new_lines[2])
        i4 = line_intersection(new_lines[1], new_lines[3])
        intersections = np.array([i1, i2, i3, i4], dtype=np.float32)
        matrix, _ = cv2.findHomography(np.float32(self.court_reference.court_conf[self.best_conf]),
                                       intersections, method=0)
        inv_matrix = cv2.invert(matrix)[1]
        self.court_warp_matrix.append(matrix)
        self.game_warp_matrix.append(inv_matrix)
        self.frame_points = intersections

        # tennis-tracking의 추가 코드
        self.pts = np.array(self.court_reference.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
        self.new_lines = cv2.perspectiveTransform(self.pts, self.court_warp_matrix[-1]).reshape(-1)

        return self.new_lines
    
    def draw_court_lines(self, frame, lines):
        # 한 frame에 테니스 코트 선을 그리는 함수
        height, width = frame.shape[:2]

        for i in range(0, len(lines), 4):
            x1, y1, x2, y2 = lines[i], lines[i+1], lines[i+2], lines[i+3]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 5)

        new_frame = cv2.resize(frame, (width, height))
        return new_frame

def line_intersection(line1, line2):
    # 2개 직선이 서로 교차하는 점을 찾음

    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates

def display_lines_on_frame(frame, horizontal=(), vertical=()):
    # 이미지에서 직선들을 표시해서 보여주는 함수

    for line in horizontal:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    for line in vertical:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    cv2.imshow('court', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    return frame

if __name__ == '__main__':
    img = cv2.imread('test.png')
    c = CourtDetector()
    c.detect(img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()