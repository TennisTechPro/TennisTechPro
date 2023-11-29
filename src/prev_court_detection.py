from sympy import Line
from itertools import combinations, product
from court_reference import CourtReference
import matplotlib.pyplot as plt

import numpy as np
import cv2

class CourtDetector:
    # Frame에서 테니스 코트를 추출하고 tracking 하는 클래스
    def __init__(self, verbose=0):
        self.verbose = verbose # 출력문 flag -> 0이면 x
        self.threshold = 200 # grayscale 임계값 (네트까지 감지)
        self.dist_tau = 3 # filtering할 때 제거할 가장자리의 픽셀 범위
        self.intensity_threshold = 40 # 주변 픽셀과의 밝기값 차이
        self.court_reference = CourtReference() # Reference Court
        self.v_width = 0 # Frame의 가로
        self.v_height = 0 # Frame의 세로
        self.frame = None # 영상의 한 프레임을 읽어서 저장
        self.gray = None # grayscale로 변환한 이미지 저장
        self.court_warp_matrix = [] # transformation matrix
        self.game_warp_matrix = [] # transformation inverse matrix
        self.court_score = 0 # transformation matrix score
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
        self.success_flag = False
        self.success_accuracy = 80
        self.success_score = 1000
        self.best_conf = None
        self.frame_points = None
        self.dist = 5

    def detect(self, frame):
        # 프레임(영상)에서 테니스 코트를 추출하는 함수

        # 각 변수 초기화
        self.frame = frame
        self.v_height, self.v_width = frame.shape[:2]

        # 이미지를 grayscale로 변환 후
        # threshold 값에 따라 이미지에서 흰 부분을 추출
        self.gray = self.binary(frame)
        #cv2.imshow('self.gray', self.gray)
        #if cv2.waitKey(0) & 0xff == 27:
        #    cv2.destroyAllWindows()

        # grayscale 이미지 Filtering
        filter_img = self.filter(self.gray)
        #cv2.imshow('filter_img', filter_img)
        #if cv2.waitKey(0) & 0xff == 27:
        #    cv2.destroyAllWindows()

        # Filtering한 이미지에서
        # 수평선, 수직선을 추출 -> Hough transform 사용
        # 현재는 공식적인 테니스 경기 영상과 같은 코트에서만 가능
        horizontal, vertical = self._detect_lines(filter_img)

        # 이미지에서 추출한 테니스 코트와
        # Reference court 사이의 transformation을 계산
        court_warp_matrix, game_warp_matrix, self.court_score = self._find_homography(horizontal, vertical)
        
        self.court_warp_matrix.append(court_warp_matrix)
        self.game_warp_matrix.append(game_warp_matrix)
        # court_accuracy = self._get_court_accuracy(0)
        # if court_accuracy > self.success_accuracy and self.court_score > self.success_score:
        #   self.success_flag = True
        # print('Court accuracy = %.2f' % court_accuracy)

        # 프레임에서 테니스 코트의 중요한 선들의 위치를 저장
        lines = self.find_lines_location()

        return lines
  

    def binary(self, frame):
        # RGB 이미지에 grayscale 적용
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 임계값보다 높은 픽셀은 255 값
        # 아닌 것은 0의 값으로 변환
        gray = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        return gray
  

    def filter(self, gray):
        # 주변 픽셀과 비교해서 이미지를 필터링함
        d_tau = self.dist_tau
        i_threshold = self.intensity_threshold
        h, w = gray.shape

        # 이미지의 가장자리에서 self.dist_tau 픽셀 만큼의 테두리를 제외
        for i in range(d_tau, h - d_tau):
            for j in range(d_tau, w - d_tau):
                if gray[i, j] == 0:
                    continue
                
                # 주변 픽셀과 self.intensity_threshold만큼 값이 차이가 나는지 확인(가로)
                if (gray[i, j] - gray[i + d_tau, j] > i_threshold and
                        gray[i, j] - gray[i - d_tau, j] > i_threshold):
                    continue
                # 세로로 확인
                if (gray[i, j] - gray[i, j + d_tau] > i_threshold and
                        gray[i, j] - gray[i, j - d_tau] > i_threshold):
                    continue

                gray[i, j] = 0
        
        return gray
    

    def _detect_lines(self, gray):
        # Hough transform을 사용해서 직선을 추출

        minLineLength = 100 # 선으로 간주할 최소 길이
        maxLineGap = 20 # 동일한 선으로 간주할 최대 간격

        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=minLineLength, maxLineGap=maxLineGap)
        lines = np.squeeze(lines)
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), [], lines)

        # 직선들의 기울기를 사용해서 수평선, 수직선 분류
        horizontal, vertical = self._classify_lines(lines)
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), horizontal, vertical)

        # 같은 선에 있는 직선들을 합병
        horizontal, vertical = self._merge_lines(horizontal, vertical)
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), horizontal, vertical)

        return horizontal, vertical
    

    def _classify_lines(self, lines):
        # Initialize counters for horizontal and vertical lines
        
        horizontal = [] # 초록색
        vertical = [] # 빨간색
        highest_vertical_y = np.inf
        lowest_vertical_y = 0

        # Classify the detected lines
        for line in lines:
            x1, y1, x2, y2 = line
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)

            # x의 변화가 y의 변화의 2배보다 크면 수평선
            if dx > 2 * dy:
                horizontal.append(line)
            else:
                vertical.append(line)
                highest_vertical_y = min(highest_vertical_y, y1, y2)
                lowest_vertical_y = max(lowest_vertical_y, y1, y2)

        # 관심 영역 내에 있는 수평선만을 검출
        clean_horizontal = []
        
        h = lowest_vertical_y - highest_vertical_y
        lowest_vertical_y += h / 15
        highest_vertical_y -= h * 2 / 15

        for line in horizontal:
            x1, y1, x2, y2 = line
            if lowest_vertical_y > y1 > highest_vertical_y and lowest_vertical_y > y1 > highest_vertical_y:
                clean_horizontal.append(line)

        return clean_horizontal, vertical

    def _merge_lines(self, horizontal_lines, vertical_lines):
        # 같은 선상에 있는 직선들을 합병하기
        
        # Merge horizontal lines
        horizontal_lines = sorted(horizontal_lines, key=lambda item: item[0])
        mask = [True] * len(horizontal_lines)
        new_horizontal_lines = []

        for i, line in enumerate(horizontal_lines):
            if mask[i]:
                for j, s_line in enumerate(horizontal_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line

                        dy = abs(y3 - y2)
                        if dy < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[0])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False
                new_horizontal_lines.append(line)

        # Merge vertical lines
        vertical_lines = sorted(vertical_lines, key=lambda item: item[1])
        xl, yl, xr, yr = (0, self.v_height * 6 / 7, self.v_width, self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_vertical_lines = []

        for i, line in enumerate(vertical_lines):
            if mask[i]:
                for j, s_line in enumerate(vertical_lines[i + 1:]):
                    if mask[i + j + 1]:
                        x1, y1, x2, y2 = line
                        x3, y3, x4, y4 = s_line
                        xi, yi = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                        xj, yj = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))

                        dx = abs(xi - xj)
                        if dx < 10:
                            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda x: x[1])
                            line = np.array([*points[0], *points[-1]])
                            mask[i + j + 1] = False

                new_vertical_lines.append(line)

        return new_horizontal_lines, new_vertical_lines
    

    def _find_homography(self, horizontal_lines, vertical_lines):
        # Reference court랑 이미지의 테니스 코트 사이의 transformation 계산

        max_score = -np.inf
        max_mat = None
        max_inv_mat = None
        
        # Loop over every pair of horizontal lines and every pair of vertical lines
        for h_pair, v_pair in product(combinations(horizontal_lines, 2), combinations(vertical_lines, 2)):
                h1, h2 = h_pair
                v1, v2 = v_pair
                
                # 2개의 직선이 교차하는 지점을 찾음
                intersections = sort_intersection_points([
                    line_intersection((h1[:2], h1[2:]), (v1[0:2], v1[2:])),
                    line_intersection((h1[:2], h1[2:]), (v2[0:2], v2[2:])),
                    line_intersection((h2[:2], h2[2:]), (v1[0:2], v1[2:])),
                    line_intersection((h2[:2], h2[2:]), (v2[0:2], v2[2:]))
                ])

                for i, configuration in self.court_reference.court_conf.items():
                    # Find transformation (3x3 행렬)
                    matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections), method=0)
                    inv_matrix = cv2.invert(matrix)[1]
                    
                    # Get transformation score
                    confi_score = self._get_confi_score(matrix)

                    if max_score < confi_score:
                        max_score = confi_score
                        max_mat = matrix
                        max_inv_mat = inv_matrix
                        self.best_conf = i

        if self.verbose:
            frame = self.frame.copy()
            court = self.add_court_overlay(frame, max_mat, (255, 0, 0))
            cv2.imshow('court', court)
            if cv2.waitKey(0) & 0xff == 27:
                cv2.destroyAllWindows()
        # print(f'Score = {max_score}')
        # print(f'Combinations tested = {k}')

        return max_mat, max_inv_mat, max_score  


    def _get_confi_score(self, matrix):
        # transformation score 계산

        # warpPerpective 함수를 사용해서 transform matrix 기반으로 court_reference를 변환
        court = cv2.warpPerspective(self.court_reference.court, matrix, self.frame.shape[1::-1])
        
        # 변환된 court_referenced에서 흰색을 1로 설정
        court[court > 0] = 1
        
        # 원본 이미지의 grayscale에서 흰색을 1로 설정
        gray = self.gray.copy()
        gray[gray > 0] = 1
        
        # court_reference와 원본 테니스 코트에서 겹치는 부분을 1로 설정한 새로운 배열 생성
        correct = court * gray
        wrong = court - correct
        
        # 겹치는 부분의 합을 계산해서 correct의 픽셀 수를 계산
        c_p = np.sum(correct)
        # 겹치지 않는 부분의 합을 계산해서 correct의 픽셀 수를 계산
        w_p = np.sum(wrong)
        
        # 변환 점수 계산 후 반환
        # 테니스 코트 영역이 얼마나 정확하게 감지되었는지를 나타내는 지표
        return c_p - 0.5 * w_p


    def add_court_overlay(self, frame, homography=None, overlay_color=(255, 255, 255), frame_num=-1):
        # 추출한 테니스 코트를 겹침

        if homography is None and len(self.court_warp_matrix) > 0 and frame_num < len(self.court_warp_matrix):
            homography = self.court_warp_matrix[frame_num]
        court = cv2.warpPerspective(self.court_reference.court, homography, frame.shape[1::-1])
        frame[court > 0, :] = overlay_color
        return frame
    

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
    
        
    def get_extra_parts_location(self, frame_num=-1):
        parts = np.array(self.court_reference.get_extra_parts(), dtype=np.float32).reshape((-1, 1, 2))
        parts = cv2.perspectiveTransform(parts, self.court_warp_matrix[frame_num]).reshape(-1)
        top_part = parts[:2]
        bottom_part = parts[2:]
        return top_part, bottom_part


    def delete_extra_parts(self, frame, frame_num=-1):
        img = frame.copy()
        top, bottom = self.get_extra_parts_location(frame_num)
        img[int(bottom[1] - 10):int(bottom[1] + 10), int(bottom[0] - 15):int(bottom[0] + 15), :] = (0, 0, 0)
        img[int(top[1] - 10):int(top[1] + 10), int(top[0] - 15):int(top[0] + 15), :] = (0, 0, 0)
        return img
    
    
    def get_warped_court(self):
        # warp된 코트를 반환

        court = cv2.warpPerspective(self.court_reference.court, self.court_warp_matrix[-1], self.frame.shape[1::-1])
        court[court > 0] = 1
        return court

    def _get_court_accuracy(self, verbose=0):
        # 추출한 테니스 코트를 겹침

        frame = self.frame.copy()
        gray = self._threshold(frame)
        gray[gray > 0] = 1
        gray = cv2.dilate(gray, np.ones((9, 9), dtype=np.uint8))

        court = self.get_warped_court()
        total_white_pixels = sum(sum(court))

        sub = court.copy()
        sub[gray == 1] = 0
        accuracy = 100 - (sum(sum(sub)) / total_white_pixels) * 100

        if verbose:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(gray, cmap='gray')
            plt.title('Grayscale frame'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 3, 2)
            plt.imshow(court, cmap='gray')
            plt.title('Projected court'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 3, 3)
            plt.imshow(sub, cmap='gray')
            plt.title('Subtraction result'), plt.xticks([]), plt.yticks([])
            plt.show()
        return accuracy

    
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

def sort_intersection_points(intersections):
    # 교차점들을 정렬 (왼쪽 상단 -> 오른쪽 하단)

    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34

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