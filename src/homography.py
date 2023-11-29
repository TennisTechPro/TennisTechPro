import cv2
import numpy as np
from scipy.spatial import distance

# transformation matrix 찾기
def get_trans_matrix(court_ref, court_conf_ind, points, refer_kps):
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
    return matrix_trans, inv_matrix_trans