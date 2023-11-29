import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad, bias=True, bn=True):
        super().__init__()
        if bn:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
                nn.ReLU()
            )

    def forward(self, x):
        return self.block(x)
    
class BallTrackNet(nn.Module):
    def __init__(self, out_channels=256, bn=True):
        super().__init__()
        self.out_channels = out_channels

        # Encoder layers
        layer_1 = ConvBlock(in_channels=9, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_4 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_5 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_6 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_7 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_8 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_9 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_10 = nn.MaxPool2d(kernel_size=2, stride=2)
        layer_11 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_12 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_13 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)

        self.encoder = nn.Sequential(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9,
                                     layer_10, layer_11, layer_12, layer_13)

        # Decoder layers
        layer_14 = nn.Upsample(scale_factor=2)
        layer_15 = ConvBlock(in_channels=512, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_16 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_17 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_18 = nn.Upsample(scale_factor=2)
        layer_19 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_20 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_21 = nn.Upsample(scale_factor=2)
        layer_22 = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_23 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
        layer_24 = ConvBlock(in_channels=64, out_channels=self.out_channels, kernel_size=3, pad=1, bias=True, bn=bn)

        self.decoder = nn.Sequential(layer_14, layer_15, layer_16, layer_17, layer_18, layer_19, layer_20, layer_21,
                                     layer_22, layer_23, layer_24)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x, testing=False):
        batch_size = x.size(0)
        features = self.encoder(x)
        scores_map = self.decoder(features)
        output = scores_map.reshape(batch_size, self.out_channels, -1)
        # output = output.permute(0, 2, 1)
        if testing:
            output = self.softmax(output)
        return output

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def inference(self, frames: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if len(frames.shape) == 3:
                frames = frames.unsqueeze(0)
            if next(self.parameters()).is_cuda:
                frames.cuda()
            # Forward pass
            output = self(frames, True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            if self.out_channels == 2:
                output *= 255
            x, y = self.get_center_ball(output)
        return x, y

    def get_center_ball(self, output):
        """
        Detect the center of the ball using Hough circle transform
        :param output: output of the network
        :return: indices of the ball`s center
        """
        output = output.reshape((360, 640))

        # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
        output = output.astype(np.uint8)

        # reshape the image size as original input image
        heatmap = cv2.resize(output, (640, 360))

        # heatmap is converted into a binary image by threshold method.
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        # find the circle in image with 2<=radius<=7
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
        # check if there have any tennis be detected
        if circles is not None:
            # if only one tennis be detected
            if len(circles) == 1:
                x = int(circles[0][0][0])
                y = int(circles[0][0][1])

                return x, y
        return None, None
    

class BallDetector:
    def __init__(self, save_state='models/tracknetV1/tracknet_pytorch.pth', out_channels=2):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load TrackNet model weights
        self.detector = BallTrackNet(out_channels=out_channels)
        saved_state_dict = torch.load(save_state, map_location=self.device)
        self.detector.load_state_dict(saved_state_dict['model_state'])
        self.detector.eval().to(self.device)

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None

        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360

        self.threshold_dist = 100
        self.xy_coordinates = np.array([[None, None], [None, None]])

        self.bounces_indices = []

    def detect_ball(self, frame):

        # Save frame dimensions
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()

        # detect only in 3 frames were given
        if self.last_frame is not None:
            # combine the frames into 1 input tensor
            frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
                                          self.model_input_width, self.model_input_height)
            frames = (torch.from_numpy(frames) / 255).to(self.device)
            # Inference (forward pass)
            x, y = self.detector.inference(frames)
            if x is not None:
                # Rescale the indices to fit frame dimensions
                x = x * (self.video_width / self.model_input_width)
                y = y * (self.video_height / self.model_input_height)

                # Check distance from previous location and remove outliers
                if self.xy_coordinates[-1][0] is not None:
                    if np.linalg.norm(np.array([x,y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                        x, y = None, None
            self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)
    
    def preprocessing_ball_coords(self):
        print('Preprocessing ball coordinates...')
        self.xy_coordinates = np.array([[None, None] if coord is None else coord for coord in self.xy_coordinates])
        ball_x, ball_y = self.xy_coordinates[:, 0], self.xy_coordinates[:, 1]
        smooth_x = signal.savgol_filter(ball_x, 3, 2)
        smooth_y = signal.savgol_filter(ball_y, 3, 2)

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


def combine_three_frames(current_frame, before_last_frame, last_frame, width, height):
    """
    Combine three frames into one input tensor for detecting the ball
    """
    # Resize and type converting for each frame
    current_frame = cv2.resize(current_frame, (width, height))
    # input must be float type
    current_frame = current_frame.astype(np.float32)

    # Resize it
    before_last_frame = cv2.resize(before_last_frame, (width, height))
    # input must be float type
    before_last_frame = before_last_frame.astype(np.float32)

    # Resize it
    last_frame = cv2.resize(last_frame, (width, height))
    # input must be float type
    last_frame = last_frame.astype(np.float32)

    # Combine three imgs to (width , height, rgb*3)
    imgs = np.concatenate((current_frame, before_last_frame, last_frame), axis=2)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    imgs = np.rollaxis(imgs, 2, 0)
    return imgs