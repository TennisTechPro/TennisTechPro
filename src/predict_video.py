import argparse
import cv2
import imutils
import time

from court_detection import CourtDetector
from detection import DetectionModel
from pose import PoseExtractor
from action_recognition import ActionRecognition
from ball_detection import BallDetector
from utils import get_video_properties

# parse parameters
parser = argparse.ArgumentParser(description='Tennis Game Analysis')

parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")
parser.add_argument("--stickman", type=int, default=0)
parser.add_argument("--minimap", type=int, default=0)
parser.add_argument("--bounce", type=int, default=0)

args = parser.parse_args()
input_video_path = args.input_video_path
output_video_path = args.output_video_path
minimap = args.minimap
bounce = args.bounce


def merge(frame, image):
    frame_h, frame_w = frame.shape[:2]

    width = frame_w // 4
    resized = imutils.resize(image, width=width)

    img_h, img_w = resized.shape[:2]
    w = frame_w - img_w

    frame[:img_h, w:] = resized

    return frame


def create_minimap(court_detector, player_detector, ball_detector, fps):
    """
    Creates top view video of the gameplay
    """
    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/minimap.mp4', fourcc, fps, (v_width, v_height))

    # Marking players location on court
    frame_num = 0
    smoothed_1, smoothed_2 = player_detector.calculate_feet_positions(court_detector)
    for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
        frame = court.copy()
        if feet_pos_1[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 15, (255, 0, 0), -1)
        if feet_pos_2[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 15, (255, 0, 0), -1)
        # frame = ball_detector.draw_ball_position_in_minimap(frame, court_detector, frame_num)
        frame_num += 1
        out.write(frame)
    out.release()


def add_minimap(output_video_path):
    video = cv2.VideoCapture('output/analysis.mp4')
    fps, length, v_width, v_height = get_video_properties(video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (v_width, v_height))
    
    print('Adding the minimap...')

    minimap_video = cv2.VideoCapture('output/minimap.mp4')
    while True:
        ret, frame = video.read()
        ret2, minimap_frame = minimap_video.read()
        if ret and ret2:
            output = merge(frame, minimap_frame)
            output_video.write(output)
        else:
            break
    video.release()
    minimap_video.release()
    output_video.release()


def process(input_video_path, output_video_path):
    # Set start time
    start_time = time.time()

    # Load videos from videos path
    video = cv2.VideoCapture(input_video_path)

    # Get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # Output Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    analysis_video = cv2.VideoWriter('output/analysis.mp4', fourcc, fps, (v_width, v_height))

    # Initialize
    court_detector = CourtDetector() # Court Detector
    player_detector = DetectionModel() # Player detector
    pose_extractor = PoseExtractor() # Pose Extractor
    # action_recognition = ActionRecognition() # Action Recognition
    ball_detector = BallDetector() # tracknet

    frame_num = 0 # frame counter
    frames = [] # Save all frame

    # First Part: Court Line Detection, Object Detection, Pose Estimation, Ball Detection
    while True:
        ret, frame = video.read()
        frame_num += 1

        if ret:
            if frame_num == 1:
                print('Start!')
                print('Detecting the court and the players...')
                print("Estimate player's pose...")
                lines = court_detector.detect(frame)
            else:
                # then track court
                lines = court_detector.track_court(frame)

            # Detect ball
            frame = ball_detector.detect_ball_in_one_frame(frame, v_width, v_height)
            
            # Detect two players
            frame = player_detector.detect_bottom_player(frame, court_detector) # frame vs frame.copy()
            frame = player_detector.detect_top_player(frame, court_detector) # frame vs frame.copy()

            # Estimate player's pose
            player1_boxes, player2_boxes = player_detector.get_boxes()
            frame = pose_extractor.extract_pose(frame, player1_boxes, player2_boxes)

            # Draw court line
            new_frame = court_detector.draw_court_lines(frame, lines)
            frames.append(new_frame)
                
            analysis_video.write(new_frame)
        else:
            break
    video.release() # Video 획득 개체를 해제
    analysis_video.release()

    # Second Part: Pose Estimation, Action Recognition


    # Third Part: Processing ball coordinates
    print('전처리 전 \n', ball_detector.xy_coordinates)
    ball_detector.remove_outliers()
    ball_detector.interpolate_coords()
    # ball_detector.preprocessing_ball_coords()
    print('전처리 후 \n', ball_detector.xy_coordinates)

    # Fourth Part: Add minimap in video
    create_minimap(court_detector, player_detector, ball_detector, fps) # minimap video를 생성
    add_minimap(output_video_path) # output video와 minimap video를 합친 하나의 video 생성
    
    # Measure processing time
    end_time = time.time()
    total_time = end_time - start_time

    print('Finished!')
    print(f'Analysis Time: {round(total_time)}s')
    print('Total Frame: ', frame_num)
    player_detector.print_counts()
    pose_extractor.print_counts()

if __name__ == '__main__':
    input_video_path = 'test/video_input6.mp4'
    output_video_path = 'output/output.mp4'
    process(input_video_path, output_video_path)