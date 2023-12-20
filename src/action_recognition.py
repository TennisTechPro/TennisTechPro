import cv2
import numpy as np
from keras.models import load_model

class ActionRecognition:
    def __init__(self):
        self.model = load_model('models/gru80_seq36.h5', compile=False)
        self.model.compile()
        self.seq_length = 36
        self.motions = ['backhand', 'forehand', 'serve/smash', 'volley']
    
    def predict_players_motion(self, frame, frame_num, p1_keypoints, p2_keypoints):
        """Sequence 길이만큼 Pose를 추정했을 때 선수들의 동작을 예측"""
        if frame_num >= self.seq_length - 1:
            # Player 1의 동작 예측
            p1_kpts = p1_keypoints[frame_num - self.seq_length + 1 : frame_num + 1]
            p1_kpts = np.array(p1_kpts).reshape(1, self.seq_length, 26)
            p1_probs = self.model.predict(p1_kpts)[0]
            
            # Player 2의 동작 예측
            p2_kpts = p2_keypoints[frame_num - self.seq_length + 1 : frame_num + 1]
            p2_kpts = np.array(p2_kpts).reshape(1, self.seq_length, 26)
            p2_probs = self.model.predict(p2_kpts)[0]
        else:
            p1_probs = np.zeros(4)
            p2_probs = np.zeros(4)

        frame = draw_probs(frame, p1_probs, 1)
        frame = draw_probs(frame, p2_probs, 2)
        return frame
    
def draw_probs(frame, probs, player):
    """각 선수들의 동작을 예측한 확률을 Dynamic Bar로 표현"""
    TEXT_ORIGIN_X = 20
    SPACE_BETWEEN_BARS = 60
    BAR_WIDTH = 40
    BAR_ORIGIN_X = 15
    BAR_HEIGHT = 200

    if player == 1:
        MARGIN_ABOVE_BAR = frame.shape[0] * 2 // 3
        y = frame.shape[0] - 100
        color = (255, 0, 255)
    elif player == 2:
        MARGIN_ABOVE_BAR = frame.shape[0] // 9
        y = int(frame.shape[0] / 2.8)
        color = (255, 255, 0)

    cv2.putText(frame, "B", (TEXT_ORIGIN_X, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=3) # Backhand
    cv2.putText(frame, "F", (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=3) # Forehand
    cv2.putText(frame, "S", (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS * 2, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=3) # Serve/Smash
    cv2.putText(frame, "V", (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS * 3, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=3) # Volley

    # Backhand
    cv2.rectangle(frame,
                    (BAR_ORIGIN_X, int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[0])),
                    (BAR_ORIGIN_X + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR),
                    color=color,
                    thickness=-1
                    )
    
    # Forehand
    cv2.rectangle(frame,
                    (BAR_ORIGIN_X + SPACE_BETWEEN_BARS, int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[1])),
                    (BAR_ORIGIN_X + SPACE_BETWEEN_BARS + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR),
                    color=color,
                    thickness=-1
                    )
    
    # Serve/Smash
    cv2.rectangle(frame,
                    (BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 2, int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[2])),
                    (BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 2 + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR),
                    color=color,
                    thickness=-1
                    )
    
    # Volley
    cv2.rectangle(frame,
                    (BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 3, int(BAR_HEIGHT + MARGIN_ABOVE_BAR - BAR_HEIGHT * probs[3])),
                    (BAR_ORIGIN_X + SPACE_BETWEEN_BARS * 3 + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR),
                    color=color,
                    thickness=-1
                    )      

    for i in range(4):
        cv2.rectangle(frame,
                (BAR_ORIGIN_X + SPACE_BETWEEN_BARS * i, int(MARGIN_ABOVE_BAR)),
                (BAR_ORIGIN_X + SPACE_BETWEEN_BARS * i + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR),
                color=(255, 255, 255),
                thickness=1
                )

    return frame