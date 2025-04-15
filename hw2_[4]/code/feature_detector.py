import numpy as np
import cv2
from matplotlib import pyplot as plt
import gc
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import cg, spsolve, spilu, LinearOperator, lgmres

class FeatureDetector:
    def __init__(
            self):
        pass

    def detect(self, im: np.ndarray):
        raise NotImplementedError()
    
    def compute(self, im: np.ndarray):
        raise NotImplementedError()

class SIFT(FeatureDetector):
    def __init__(
            self):
        super().__init__()
        self.s = 3 # scale per octave
        self.sigma = 1.6 # of gaussian kernels
        
    def detect(self, im):
        # compute octaves
        gaussian_octaves = self.compute_gaussian_octaves(im)
        DoG_octaves = self.compute_DoG_octaves(gaussian_octaves)
        
        # locate local extrema
        keypoint_candidates = self.compute_keypoint_candidates(DoG_octaves)

        adjusted_keypoint_candidates, keypoint_offsets = self.compute_keypoint_offsets(keypoint_candidates, DoG_octaves)
        return []
    
    def compute_gaussian_octaves(self, im):
        octaves = []
        cur_base_im = cv2.GaussianBlur(im, (0, 0), self.sigma)
        k = 2 ** (1 / self.s)

        while min(cur_base_im.shape[0], cur_base_im.shape[1]) >= 16:
            cur_octave = [cur_base_im]

            for i in range(1, self.s + 3):
                cur_octave.append(cv2.GaussianBlur(cur_base_im, (0, 0), (k ** i) * self.sigma))
            
            octaves.append(cur_octave)
            cur_base_im = cur_octave[-3]
            cur_base_im = cv2.resize(cur_base_im, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            
        return octaves
    
    def compute_DoG_octaves(self, gaussian_octaves):
        octaves = []
        
        for cur_gaussian_octave in gaussian_octaves:
            cur_octave = []

            for i in range(self.s + 2):
                cur_octave.append(cur_gaussian_octave[i + 1] - cur_gaussian_octave[i])
            
            octaves.append(cur_octave)
        
        return octaves
    
    def compute_keypoint_candidates(self, DoG_octaves):
        offsets = []
        for dk in range(-1, 2):
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if dk == 0 and di == 0 and dj == 0:
                        continue
                    offsets.append((dk, di, dj))

        keypoint_candidates = []
        for octave_idx, cur_octave in enumerate(DoG_octaves):
            for k in range(1, self.s + 1):
                for i in range(1, cur_octave[k].shape[0] - 1):
                    for j in range(1, cur_octave[k].shape[1] - 1):
                        sign = 0
                        found_extrema = True
                        for dk, di, dj in offsets:
                            difference = cur_octave[k][i][j].astype(np.int32) - cur_octave[k + dk][i + di][j + dj]
                            if difference == 0:
                                found_extrema = False
                                break
                            else:
                                if sign == 0:
                                    sign = np.sign(difference)
                                else:
                                    cur_sign = np.sign(difference)
                                    if cur_sign != sign:
                                        found_extrema = False
                                        break

                        if found_extrema:
                            keypoint_candidates.append((octave_idx, i, j, k))
                        # neighbors = []
                        # for dk, di, dj in offsets:
                        #     neighbors.append(cur_octave[k + dk][i + di][j + dj])
                        
                        # if np.all(cur_octave[k][i][j] > neighbors) or np.all(cur_octave[k][i][j] < neighbors):
                        #     keypoint_candidates.append((octave_idx, i, j, k))
        # print(len(keypoint_candidates))
        return keypoint_candidates
    
    def compute_keypoint_offsets(self, keypoint_candidates, DoG_octaves):
        adjusted_keypoint_candidates = []
        offsets = []

        for octave_idx, i, j, k in keypoint_candidates:
            D = DoG_octaves[octave_idx]

            x = j
            y = i
            s = k

            D_x = (1.0 * D[s][y][x + 1] - D[s][y][x - 1]) / 2.0
            D_y = (1.0 * D[s][y + 1][x] - D[s][y - 1][x]) / 2.0
            D_s = (1.0 * D[s + 1][y][x + 1] - D[s - 1][y][x]) / 2.0

            D_xx = D[s][y][x + 1] - 2.0 * D[s][y][x] + D[s][y][x - 1]
            D_yy = D[s][y + 1][x] - 2.0 * D[s][y][x] + D[s][y - 1][x]
            D_ss = D[s + 1][y][x] - 2.0 * D[s][y][x] + D[s - 1][y][x]

            D_xy = (1.0 * D[s][y - 1][x - 1] - D[s][y + 1][x - 1] - D[s][y - 1][x + 1] + D[s][y + 1][x + 1]) / 4.0
            D_xs = (1.0 * D[s - 1][y][x - 1] - D[s + 1][y][x - 1] - D[s - 1][y][x + 1] + D[s + 1][y][x + 1]) / 4.0
            D_ys = (1.0 * D[s - 1][y - 1][x] - D[s + 1][y - 1][x] - D[s - 1][y + 1][x] + D[s + 1][y + 1][x]) / 4.0

            gradient = np.array([D_x, D_y, D_s])
            
            Hessian = np.array([
                [D_xx, D_xy, D_xs],
                [D_xy, D_yy, D_ys],
                [D_xs, D_ys, D_ss]
            ])

            Hessian_inverse = np.linalg.inv(Hessian)

            offset = -1.0 * Hessian_inverse.dot(gradient)
            # print(offset)
        return adjusted_keypoint_candidates, offsets