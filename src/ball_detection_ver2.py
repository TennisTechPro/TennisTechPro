import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from tensorflow.keras.utils import array_to_img, img_to_array
from keras.models import load_model
from keras.layers import *
import keras.backend as K
from tracknetV2 import TrackNet

def combine_three_frames(current_frame, before_last_frame, last_frame):
    unit = []

    # Adjust BGR format (cv2) to RGB format (PIL)
    x1 = last_frame[...,::-1]
    x2 = before_last_frame[...,::-1]
    x3 = current_frame[...,::-1]

    # Convert np arrays to PIL Images
    x1 = array_to_img(x1)
    x2 = array_to_img(x2)
    x3 = array_to_img(x3)

    # Resize the images
    x1 = x1.resize(size=(512, 288))
    x2 = x2.resize(size=(512, 288))
    x3 = x3.resize(size=(512, 288))

    # Convert images to np arrays and adjust to channels first
    x1 = np.moveaxis(img_to_array(x1), -1, 0)
    x2 = np.moveaxis(img_to_array(x2), -1, 0)
    x3 = np.moveaxis(img_to_array(x3), -1, 0)

    # Create data
    unit.append(x1[0])
    unit.append(x1[1])
    unit.append(x1[2])
    unit.append(x2[0])
    unit.append(x2[1])
    unit.append(x2[2])
    unit.append(x3[0])
    unit.append(x3[1])
    unit.append(x3[2])
    
    unit = np.asarray(unit)
    unit = unit.reshape((1, 9, 288, 512))
    unit = unit.astype('float32')
    unit /= 255
    
    return unit

def custom_loss(y_true, y_pred):
    loss = (-1) * (K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
    return K.mean(loss)

class BallDetector:
    def __init__(self):
        self.tracknet = TrackNet()
        self.model = load_model('models/tracknetV2/model_3in1', custom_objects={'custom_loss':custom_loss})
        self.xy_coordinates = [None, None]
        self.last_frame = None
        self.before_last_frame = None
        self.current_frame = None

    def detect_ball(self, frame):
        print('Detecting the ball...')

        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()

        ratio = frame.shape[0] / 288

        if self.last_frame is not None:
            imgs = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame)
            
            y_pred = self.model.predict(imgs, batch_size=1)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype('float32')
            h_pred = y_pred[0] * 255
            h_pred = h_pred.astype('uint8')
            
            if np.amax(h_pred) <= 0:
                self.xy_coordinates.append(None)
            else:
                cnts, _ = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for i in range(len(rects)):
                    area = rects[i][2] * rects[i][3]
                    if area > max_area:
                        max_area_idx = i
                        max_area = area
                target = rects[max_area_idx]
                cx_pred, cy_pred = int(ratio * (target[0] + target[2] / 2)), int(ratio * (target[1] + target[3] / 2))
                self.xy_coordinates.append([cx_pred, cy_pred])

    def preprocessing_ball_coords(self):
        print('Preprocessing ball coordinates...')
        self.xy_coordinates = np.array([[None, None] if coord is None else coord for coord in self.xy_coordinates])
        ball_x, ball_y = self.xy_coordinates[:, 0], self.xy_coordinates[:, 1]
        smooth_x = signal.savgol_filter(ball_x, 5, 2)
        smooth_y = signal.savgol_filter(ball_y, 5, 2)

        # interpolation
        x = np.arange(0, len(smooth_y))
        indices = [i for i, val in enumerate(smooth_y) if np.isnan(val)]
        x = np.delete(x, indices)
        y1 = np.delete(smooth_y, indices)
        y2 = np.delete(smooth_x, indices)

        # Sort
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y1_sorted = y1[sorted_indices]
        y2_sorted = y2[sorted_indices]

        ball_f2_y = interp1d(x_sorted, y1_sorted, kind='cubic', fill_value="extrapolate")
        ball_f2_x = interp1d(x_sorted, y2_sorted, kind='cubic', fill_value="extrapolate")

        # ball_f2_y = interp1d(x, y1, kind='cubic', fill_value="extrapolate")
        # ball_f2_x = interp1d(x, y2, kind='cubic', fill_value="extrapolate")

        xnew = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)

        self.xy_coordinates[:, 0] = ball_f2_x(xnew)
        self.xy_coordinates[:, 1] = ball_f2_y(xnew)

    def draw_ball_position_in_minimap(self, frame, court_detector, frame_num):
        """
        Calculate the ball position of both players using the inverse transformation of the court and the x, y positions
        """
        inv_mats = court_detector.game_warp_matrix[frame_num]
        coord = self.xy_coordinates[frame_num]
        img = frame.copy()

        # Ball locations
        if coord is not None:
            p = np.array(coord, dtype='float64')
            ball_pos = np.array([p[0], p[1]]).reshape((1, 1, 2))
            transformed = cv2.perspectiveTransform(ball_pos, inv_mats)[0][0].astype('int64')
            img = cv2.circle(img, (transformed[0], transformed[1]), 20, (0, 255, 255), -1)

        return img