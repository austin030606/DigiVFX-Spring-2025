import numpy as np
import cv2
from matplotlib import pyplot as plt
import gc
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import cg, spsolve, spilu, LinearOperator, lgmres
from scipy.ndimage import maximum_filter, minimum_filter

class Keypoint:
    def __init__(
            self,
            octave_idx = None, 
            y = None, 
            x = None, 
            s = None, 
            orientation = None):
        self.octave_idx = octave_idx
        self.y = y
        self.x = x
        self.s = s
        self.orientation = orientation

        if octave_idx != None:
            coord_scale = 2 ** (octave_idx - 1)
            self.pt = np.array([x * coord_scale, y * coord_scale])
        else:
            self.pt = np.array([x, y])

class FeatureDetector:
    def __init__(
            self):
        pass

    def detect(self, im: np.ndarray):
        raise NotImplementedError()
    
    def compute(self, keypoints, im: np.ndarray):
        raise NotImplementedError()
    
    def detectAndCompute(self, im):
        keypoints = self.detect(im)
        print("keypoints found")
        descriptors, _ = self.compute(keypoints, im)
        print("descriptors calculated")

        return keypoints, descriptors

class SIFT(FeatureDetector):
    def __init__(
            self):
        super().__init__()
        self.s = 3 # scale per octave
        self.sigma = 1.6 # for gaussian kernels
        self.max_iterations = 8
        self.contrast_threshold = 0.04
        self.r = 10
        self.extremum_values = []
    
    def compute(self, keypoints, im):
        if im.ndim == 3 and im.shape[2] >= 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # compute octaves
        im_f = im.copy().astype(np.float32) / 255.0
        im_f = cv2.resize(im_f, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gaussian_octaves, grad_x_octaves, grad_y_octaves = self.compute_gaussian_octaves(im_f)
        if len(keypoints) > 0 and keypoints[0].orientation == None:
            keypoint_tuples = []
            for kp in keypoints:
                # (octave_idx, y, x, s)
                keypoint_tuples.append((kp.octave_idx, kp.y, kp.x, kp.s))
            keypoints = self.compute_oriented_keypoints(keypoint_tuples, gaussian_octaves, grad_x_octaves, grad_y_octaves)
        descriptors = []
        # (octave_idx, y, x, s, orientation)
        for kp in keypoints:
            octave_idx = kp.octave_idx
            y = kp.y
            x = kp.x
            s = kp.s

            # scale = 2 ** (octave_idx - 1)
            y = int(np.round(y))
            x = int(np.round(x))
            s = int(np.round(s))

            grad_x_L = grad_x_octaves[octave_idx][s]
            grad_y_L = grad_y_octaves[octave_idx][s]
            descriptors.append(self.compute_descriptor(kp, grad_x_L, grad_y_L))
        
        return descriptors, keypoints
    
    def compute_descriptor(self, kp, grad_x_L, grad_y_L):
        theta = np.deg2rad(kp.orientation)
        # grid offsets
        offsets = np.linspace(-7.5, +7.5, 16)
        u, v = np.meshgrid(offsets, offsets) # (16,16) * 2
        gaussian_weights = np.exp(-((u * u + v * v)/(2 * 8 * 8))).ravel()

        R = np.array([[ np.cos(theta), -np.sin(theta)],
                      [ np.sin(theta),  np.cos(theta)]])
        uv = np.stack([u.ravel(), v.ravel()], axis=1) # (256,2)
        rot_uv = uv.dot(R.T).reshape(16, 16, 2) # (16,16,2)
        y = kp.y
        x = kp.x
        map_x = (rot_uv[...,0] + x).astype(np.float32)        # (16,16)
        map_y = (rot_uv[...,1] + y).astype(np.float32)        # (16,16)
        
        # SLOW
        # gradient_angles = np.arctan2(grad_y_L, grad_x_L)
        # relative_gradient_angles = gradient_angles - theta
        # relative_gradient_angles = (relative_gradient_angles + 2 * np.pi) % (2 * np.pi)
        # gradient_magnitudes = np.hypot(grad_x_L, grad_y_L) 


        # sampled_gradient_magnitudes = cv2.remap(gradient_magnitudes, map_x, map_y, interpolation=cv2.INTER_LINEAR).ravel()
        # sampled_relative_gradient_angles = cv2.remap(relative_gradient_angles, map_x, map_y, interpolation=cv2.INTER_LINEAR).ravel()
        # sampled_relative_gradient_angles = (sampled_relative_gradient_angles + 2 * np.pi) % (2 * np.pi)

        sampled_grad_x_L = cv2.remap(grad_x_L, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        sampled_grad_y_L = cv2.remap(grad_y_L, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        sampled_gradient_magnitudes = np.hypot(sampled_grad_x_L, sampled_grad_y_L).ravel()
        sampled_gradient_angles = np.arctan2(sampled_grad_y_L, sampled_grad_x_L)
        sampled_relative_gradient_angles = sampled_gradient_angles - theta
        sampled_relative_gradient_angles = (sampled_relative_gradient_angles + 2 * np.pi) % (2 * np.pi)

        kernel = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]], dtype=np.float32) / 256.0
        # kernel = np.array([[1, 2, 1],
        #                    [2, 4, 2],
        #                    [1, 2, 1]], dtype=np.float32) / 16.0
        sampled_relative_gradient_angles = cv2.filter2D(
            sampled_relative_gradient_angles,
            ddepth=-1,
            kernel=kernel,
            borderType=cv2.BORDER_REFLECT
        ).ravel()


        bin_ids = ((sampled_relative_gradient_angles) / (np.pi / 4)).astype(int) % 8
        cell_id = np.arange(256)
        cell_id = 4 * (cell_id // 64) + (cell_id % 16) // 4
        bin_ids += cell_id * 8
        
        feature_vector = np.bincount(
            bin_ids,
            weights = sampled_gradient_magnitudes * gaussian_weights,
            minlength = 128
        )

        # SLOW
        # orientation_histograms = np.zeros((4, 4, 8))
        # for i in range(16):
        #     for j in range(16):
        #         h_i = i // 4
        #         h_j = j // 4
        #         hist_idx = int((sampled_relative_gradient_angles[i * 16 + j]) / (np.pi / 4)) % 8

        #         orientation_histograms[h_i][h_j][hist_idx] += sampled_gradient_magnitudes[i * 16 + j] * gaussian_weights[i * 16 + j]
        # # print(orientation_histogram - orientation_histogram.flatten())
        # feature_vector = orientation_histograms.flatten()
        
        feature_vector /= np.linalg.norm(feature_vector)
        feature_vector = np.clip(feature_vector, feature_vector.min(), 0.2)
        feature_vector /= np.linalg.norm(feature_vector)

        return feature_vector.tolist()

    def detect(self, im):
        # for debugging
        # (h, w) = im.shape[:2]
        # center = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D(center, 37.5, 1.0)
        # im = cv2.warpAffine(im, M, (w, h))
        if im.ndim == 3 and im.shape[2] >= 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # compute octaves
        im_f = im.copy().astype(np.float32) / 255.0
        im_f = cv2.resize(im_f, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gaussian_octaves, grad_x_octaves, grad_y_octaves = self.compute_gaussian_octaves(im_f)
        DoG_octaves = self.compute_DoG_octaves(gaussian_octaves)
        
        # locate local extrema
        keypoint_candidates = self.compute_keypoint_candidates(DoG_octaves)

        adjusted_keypoint_candidates, _ = self.compute_accurate_keypoints(keypoint_candidates, DoG_octaves)
        keypoints_with_orientation = self.compute_oriented_keypoints(adjusted_keypoint_candidates, gaussian_octaves, grad_x_octaves, grad_y_octaves)
        
        # im_with_keypoints = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR)
        # for kp in keypoints_with_orientation:
        #     # (octave_idx, y, x, s, orientation)
        #     octave_idx = kp.octave_idx
        #     y = kp.y
        #     x = kp.x
        #     coord_scale = 2 ** (octave_idx - 1)
        #     y = int(np.round(y * coord_scale))
        #     x = int(np.round(x * coord_scale))
        #     cv2.circle(im_with_keypoints, (x, y), 2, (0, 255, 0), 1)

        #     angle = kp.orientation
        #     end_x = int(x + 5 * coord_scale * np.cos(np.deg2rad(angle)))
        #     end_y = int(y + 5 * coord_scale * np.sin(np.deg2rad(angle)))
        #     cv2.arrowedLine(im_with_keypoints, (x, y), (end_x, end_y), (255, 0, 0), 1, tipLength=0.3)
            
        # # M = cv2.getRotationMatrix2D(center, -37.5, 1.0)
        # # im_with_keypoints = cv2.warpAffine(im_with_keypoints, M, (w, h))
        # cv2.imshow('Keypoints', im_with_keypoints)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # values = np.array(self.extremum_values)
        # print(values.min(), values.max(), values.mean())
        # plt.hist(values)
        # plt.show()
        return keypoints_with_orientation
    
    def compute_gaussian_octaves(self, im):
        octaves = []
        grad_x_octaves = []
        grad_y_octaves = []
        sigma_init = np.sqrt(self.sigma * self.sigma - 1.0 * 1.0)
        cur_base_im = cv2.GaussianBlur(im, (0, 0), sigma_init)
        cur_base_sigma = self.sigma
        # cur_base_im = im
        k = 2 ** (1 / self.s)

        gradx_kernel = np.array([[0,0,0],
                                 [-1,0,1],
                                 [0,0,0]])
        grady_kernel = np.array([[0,-1,0],
                                 [0,0,0],
                                 [0,1,0]])

        while min(cur_base_im.shape[0], cur_base_im.shape[1]) >= 16:
            cur_octave = [cur_base_im]
            cur_grad_x_octave = [cv2.filter2D(cur_base_im, -1, gradx_kernel, borderType=cv2.BORDER_REFLECT )]
            cur_grad_y_octave = [cv2.filter2D(cur_base_im, -1, grady_kernel, borderType=cv2.BORDER_REFLECT )]
            
            for i in range(1, self.s + 3):
                cur_sigma = np.sqrt(((k ** i) * cur_base_sigma) ** 2 - (cur_base_sigma ** 2))
                cur_blurred_im = cv2.GaussianBlur(cur_base_im, (0, 0), sigmaX=cur_sigma)
                cur_octave.append(cur_blurred_im)
                cur_grad_x_octave.append(cv2.filter2D(cur_blurred_im, -1, gradx_kernel, borderType=cv2.BORDER_REFLECT ))
                cur_grad_y_octave.append(cv2.filter2D(cur_blurred_im, -1, grady_kernel, borderType=cv2.BORDER_REFLECT ))
                
            octaves.append(cur_octave)
            grad_x_octaves.append(cur_grad_x_octave)
            grad_y_octaves.append(cur_grad_y_octave)
            cur_base_im = cur_octave[-3]
            cur_base_im = cv2.resize(cur_base_im, (cur_base_im.shape[1] // 2, cur_base_im.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
        return octaves, grad_x_octaves, grad_y_octaves
    
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
            footprint = np.ones((3, 3, 3), dtype=bool)
            footprint[1, 1, 1] = False

            # for maxima: compare each voxel to the max of its 26 neighbors
            neigh_max = maximum_filter(
                np.array(cur_octave),
                footprint=footprint,
                mode='nearest'
            )
            # for minima: compare each voxel to the min of its 26 neighbors
            neigh_min = minimum_filter(
                np.array(cur_octave),
                footprint=footprint,
                mode='nearest'
            )
            max_coords = np.stack(np.where(np.array(cur_octave) > neigh_max), axis=-1)
            min_coords = np.stack(np.where(np.array(cur_octave) < neigh_min), axis=-1)
            for coord in max_coords:
                keypoint_candidates.append((octave_idx, coord[1], coord[2], coord[0]))
            for coord in min_coords:
                keypoint_candidates.append((octave_idx, coord[1], coord[2], coord[0]))
            # SLOW
            # cnt = 0
            # for k in range(1, self.s + 1):
            #     for i in range(1, cur_octave[k].shape[0] - 1):
            #         for j in range(1, cur_octave[k].shape[1] - 1):
            #             sign = 0
            #             found_extrema = True
            #             for dk, di, dj in offsets:
            #                 difference = cur_octave[k][i][j] - cur_octave[k + dk][i + di][j + dj]
            #                 if abs(difference) < 1e-12:
            #                     found_extrema = False
            #                     break
            #                     # pass
            #                 else:
            #                     if sign == 0:
            #                         sign = np.sign(difference)
            #                     else:
            #                         cur_sign = np.sign(difference)
            #                         if cur_sign != sign:
            #                             found_extrema = False
            #                             break

            #             if found_extrema:
            #                 cnt += 1
            #                 keypoint_candidates.append((octave_idx, i, j, k))
            # print(cnt)
            # for row in max_coords:
            #     found_same = False
            #     for kp in keypoint_candidates:
            #         if row[0] == kp[3] and row[1] == kp[1] and row[2] == kp[2] and octave_idx == kp[0]:
            #             found_same = True
            #             break
            #     if not found_same:
            #         print(row, "has no match")
            # print('')
            # for row in min_coords:
            #     found_same = False
            #     for kp in keypoint_candidates:
            #         if row[0] == kp[3] and row[1] == kp[1] and row[2] == kp[2] and octave_idx == kp[0]:
            #             found_same = True
            #             break
            #     if not found_same:
            #         print(row, "has no match")
        # print(len(keypoint_candidates))
        return keypoint_candidates
    
    def compute_accurate_keypoints(self, keypoint_candidates, DoG_octaves):
        adjusted_keypoint_candidates = []
        offsets = []

        for octave_idx, i, j, k in keypoint_candidates:
            found_valid_keypoint_candidate = False
            x = j
            y = i
            s = k
            offset = np.zeros(3)
            D = DoG_octaves[octave_idx]
            for iteration in range(self.max_iterations):
                D_x = (D[s][y][x + 1] - D[s][y][x - 1]) / 2.0
                D_y = (D[s][y + 1][x] - D[s][y - 1][x]) / 2.0
                D_s = (D[s + 1][y][x] - D[s - 1][y][x]) / 2.0

                D_xx = D[s][y][x + 1] - 2.0 * D[s][y][x] + D[s][y][x - 1]
                D_yy = D[s][y + 1][x] - 2.0 * D[s][y][x] + D[s][y - 1][x]
                D_ss = D[s + 1][y][x] - 2.0 * D[s][y][x] + D[s - 1][y][x]

                D_xy = (D[s][y - 1][x - 1] - D[s][y + 1][x - 1] - D[s][y - 1][x + 1] + D[s][y + 1][x + 1]) / 4.0
                D_xs = (D[s - 1][y][x - 1] - D[s + 1][y][x - 1] - D[s - 1][y][x + 1] + D[s + 1][y][x + 1]) / 4.0
                D_ys = (D[s - 1][y - 1][x] - D[s + 1][y - 1][x] - D[s - 1][y + 1][x] + D[s + 1][y + 1][x]) / 4.0

                gradient = np.array([D_x, D_y, D_s])
                
                Hessian = np.array([
                    [D_xx, D_xy, D_xs],
                    [D_xy, D_yy, D_ys],
                    [D_xs, D_ys, D_ss]
                ])

                det = np.linalg.det(Hessian)
                if abs(det) < 1e-12:
                    break
                Hessian_inverse = np.linalg.inv(Hessian)

                offset = -1.0 * Hessian_inverse.dot(gradient)

                if np.sum(np.abs(offset) > 0.5) == 0:
                    found_valid_keypoint_candidate = True
                    break
                else:
                    if offset[0] > 0.5:
                        x += 1
                    elif offset[0] < -0.5:
                        x -= 1
                    
                    if offset[1] > 0.5:
                        y += 1
                    elif offset[1] < -0.5:
                        y -= 1
                    
                    if offset[2] > 0.5:
                        s += 1
                    elif offset[2] < -0.5:
                        s -= 1

                    # x += int(np.round(offset[0]))
                    # y += int(np.round(offset[1]))
                    # s += int(np.round(offset[2]))

                    if x < 1 or x > D[0].shape[1] - 2 or y < 1 or y > D[0].shape[0] - 2 or s < 1 or s > self.s:
                        break
            
            if found_valid_keypoint_candidate:
                if not self.is_unstable_or_an_edge(x, y, s, offset, D):
                    # adjusted_keypoint_candidates.append((octave_idx, y, x, s))
                    adjusted_keypoint_candidates.append((octave_idx, y + offset[1], x + offset[0], s + offset[2]))
                    offsets.append(offset)
            else:
                # print(f"failed to find valid keypoint candidate after {self.max_iterations} iterations", offset)
                # print(f"{np.sum(np.abs(offset) > 0.5)} dimension > 0.5", offset)
                pass
        # print(len(adjusted_keypoint_candidates))
        # print(len(keypoint_candidates))
        return adjusted_keypoint_candidates, offsets
    
    def is_unstable_or_an_edge(self, x, y, s, offset, D):
        D_x = (D[s][y][x + 1] - D[s][y][x - 1]) / 2.0
        D_y = (D[s][y + 1][x] - D[s][y - 1][x]) / 2.0
        D_s = (D[s + 1][y][x] - D[s - 1][y][x]) / 2.0
        gradient = np.array([D_x, D_y, D_s])
        extremum = D[s][y][x] + 0.5 * gradient.dot(offset)

        D_xx = D[s][y][x + 1] - 2.0 * D[s][y][x] + D[s][y][x - 1]
        D_yy = D[s][y + 1][x] - 2.0 * D[s][y][x] + D[s][y - 1][x]
        D_xy = (D[s][y - 1][x - 1] - D[s][y + 1][x - 1] - D[s][y - 1][x + 1] + D[s][y + 1][x + 1]) / 4.0

        Tr_Hessian = D_xx + D_yy
        Det_Hessian = D_xx * D_yy - D_xy * D_xy
        self.extremum_values.append(np.abs(extremum))
        return np.abs(extremum) * self.s < self.contrast_threshold or ((Tr_Hessian * Tr_Hessian) / Det_Hessian) >= ((self.r + 1) ** 2) / self.r
    
    def compute_oriented_keypoints(self, adjusted_keypoint_candidates, gaussian_octaves, grad_x_octaves, grad_y_octaves):
        keypoints_with_orientation = []

        # (octave_idx, y, x, s)
        for keypoint in adjusted_keypoint_candidates:
            octave_idx = keypoint[0]
            y = keypoint[1]
            x = keypoint[2]
            s = keypoint[3]

            # scale = 2 ** (octave_idx - 1)
            y = int(np.round(y))
            x = int(np.round(x))
            s = int(np.round(s))

            L = gaussian_octaves[octave_idx][s]
            grad_x_L = grad_x_octaves[octave_idx][s]
            grad_y_L = grad_y_octaves[octave_idx][s]
            scale = self.sigma * (2 ** (s / self.s))
            cur_sigma = 1.5 * scale

            window_radius = int(np.round(4 * cur_sigma))
            dxs, dys = np.meshgrid(np.arange(-window_radius, window_radius + 1), np.arange(-window_radius, window_radius + 1))
            xs = x + dxs
            ys = y + dys
            valid = (xs > 1) & (xs < L.shape[1] - 1) & (ys > 1) & (ys < L.shape[0] - 1)

            valid_xs = xs[valid]
            valid_ys = ys[valid]
            valid_dxs = dxs[valid]
            valid_dys = dys[valid]

            valid_grad_x = grad_x_L[valid_ys, valid_xs]
            valid_grad_y = grad_y_L[valid_ys, valid_xs]
            valid_magnitudes = np.hypot(valid_grad_x, valid_grad_y)
            angles = (np.degrees(np.arctan2(valid_grad_y, valid_grad_x)) + 360) % 360
            bin_ids = (np.round(angles / 10)).astype(int)

            # gaussian weights
            weights = np.exp(-(valid_dxs**2 + valid_dys**2)/(2*cur_sigma**2))
            weights /= (2 * np.pi * cur_sigma**2)

            # build your 36‑bin histogram in one call
            orientation_hist = np.bincount(
                bin_ids,
                weights=valid_magnitudes * weights,
                minlength=36
            )

            hist_img = orientation_hist[None, :]
            pad = 2
            padded = cv2.copyMakeBorder(
                hist_img,
                top=0, bottom=0, left=pad, right=pad,
                borderType=cv2.BORDER_WRAP
            )
            kernel = np.array([[1, 4, 6, 4, 1]], dtype=np.float32) / 16.0
            smoothed_padded = cv2.filter2D(
                padded,
                ddepth=-1,
                kernel=kernel,
                borderType=cv2.BORDER_REFLECT
            )
            orientation_hist = smoothed_padded[0, pad:-pad]


            # SLOW
            # orientation_hist = np.zeros(36)
            # for dx in range(-window_radius, window_radius + 1):
            #     for dy in range(-window_radius, window_radius + 1):
            #         cur_x = x + dx
            #         cur_y = y + dy
            #         if cur_x > 1 and cur_x < L.shape[1] - 1 and cur_y > 1 and cur_y < L.shape[0] - 1:
            #             # gradient_magnitude = np.sqrt((L[cur_y][cur_x + 1] - L[cur_y][cur_x - 1]) ** 2 + (L[cur_y + 1][cur_x] - L[cur_y - 1][cur_x]) ** 2)
            #             gradient_magnitude = np.sqrt(grad_x_L[cur_y][cur_x] ** 2 + grad_y_L[cur_y][cur_x] ** 2)
            #             # theta = np.arctan2((L[cur_y + 1][cur_x] - L[cur_y - 1][cur_x]), (L[cur_y][cur_x + 1] - L[cur_y][cur_x - 1]))
            #             theta = np.arctan2(grad_y_L[cur_y][cur_x], grad_x_L[cur_y][cur_x])
            #             theta_degrees = theta * (180 / np.pi)
            #             if theta_degrees < 0:
            #                 theta_degrees += 360
            #             bin_id = int(theta_degrees // 10)
            #             gaussian_weight = (1 / (2 * np.pi * (cur_sigma ** 2))) * np.exp((-(dx * dx + dy * dy)) / (2 * (cur_sigma ** 2)))
            #             orientation_hist[bin_id] += gaussian_weight * gradient_magnitude
            # # print(hist)
            # # print(orientation_hist)
            # # print((hist - orientation_hist) < 1e-9)
            
            max_orientation_idx = np.argmax(orientation_hist)
            max_orientation = orientation_hist[max_orientation_idx]

            peak_indices = np.nonzero(orientation_hist >= max_orientation * 0.8)[0]
            for idx in peak_indices.tolist():
                h_prev = orientation_hist[(idx - 1) % 36]
                h_cur  = orientation_hist[idx]
                h_next = orientation_hist[(idx + 1) % 36]
                if h_cur <= h_prev or h_cur <= h_next:
                    continue
                delta = 0 if (h_prev - 2 * h_cur + h_next) == 0 else (h_prev - h_next) / (2 * (h_prev - 2 * h_cur + h_next))
                keypoint_orientation = ((idx + delta) % 36) * (360.0 / 36)
                
                keypoints_with_orientation.append(Keypoint(keypoint[0], keypoint[1], keypoint[2], keypoint[3], keypoint_orientation))
            # print(f"{octave_idx} {s}")
        # print(len(keypoints_with_orientation))
        return keypoints_with_orientation
    

class HarrisCornerDetector(FeatureDetector):
    def __init__(
            self):
        pass
        self.sigma1 = 1.0
        self.sigma2 = 3.0
        self.k = 0.04
        self.threshold_percentage = 0.001

    def detect(self, im: np.ndarray):
        if im.ndim == 3 and im.shape[2] >= 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # compute octaves
        im_f = im.copy().astype(np.float32)
        im_f_blurred = cv2.GaussianBlur(im_f, (0, 0), sigmaX=self.sigma1)
        gradx_kernel = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]) / 8.0
        grady_kernel = np.array([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]]) / 8.0
        
        I_x = cv2.filter2D(im_f_blurred, -1, gradx_kernel, borderType=cv2.BORDER_REFLECT)
        I_y = cv2.filter2D(im_f_blurred, -1, grady_kernel, borderType=cv2.BORDER_REFLECT)

        I_x2 = I_x * I_x
        I_y2 = I_y * I_y
        I_xy = I_x * I_y


        S_x2 = cv2.GaussianBlur(I_x2, (0, 0), sigmaX=self.sigma2)
        S_y2 = cv2.GaussianBlur(I_y2, (0, 0), sigmaX=self.sigma2)
        S_xy = cv2.GaussianBlur(I_xy, (0, 0), sigmaX=self.sigma2)

        det_M = S_x2 * S_y2 - S_xy * S_xy
        trace_M = S_x2 + S_y2

        R = det_M - self.k * trace_M * trace_M

        footprint = np.ones((3, 3), dtype=bool)
        footprint[1, 1] = False

        
        threshold = R.max() * self.threshold_percentage
        R_thresh = np.zeros_like(R)
        mask = (R > threshold)
        R_thresh[mask] = R[mask]
        
        neigh_max = maximum_filter(R_thresh, footprint=footprint)

        max_coords = np.stack(np.where(R_thresh > neigh_max), axis=-1)

        corners = []
        for coord in max_coords:
            # print(coord)
            corners.append(Keypoint(x=coord[1], y=coord[0], octave_idx=1, s=0))

        # im_with_keypoints = cv2.cvtColor(im.copy(), cv2.COLOR_GRAY2BGR)
        # for kp in corners:
        #     # (octave_idx, y, x, s, orientation)
        #     y = kp.y
        #     x = kp.x
        #     cv2.circle(im_with_keypoints, (x, y), 2, (0, 255, 0), 1)

        # cv2.imshow('Keypoints', im_with_keypoints)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # plt.figure(figsize=(8,6))
        # plt.imshow(R, interpolation='nearest')
        # plt.colorbar(label='Corner response R')
        # plt.title('Harris Corner Response Heatmap')
        # plt.xlabel('x pixel')
        # plt.ylabel('y pixel')
        # plt.show()

        return corners
    
    def compute(self, keypoints, im: np.ndarray):
        raise NotImplementedError("Harris corner detector does not compute descriptors")
    

class PCA_SIFT(SIFT):
    def __init__(
            self):
        super().__init__()
        self.n = 20

    def compute(self, keypoints, im):
        if im.ndim == 3 and im.shape[2] >= 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # compute octaves
        im_f = im.copy().astype(np.float32) / 255.0
        im_f = cv2.resize(im_f, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gaussian_octaves, grad_x_octaves, grad_y_octaves = self.compute_gaussian_octaves(im_f)
        if len(keypoints) > 0 and keypoints[0].orientation == None:
            keypoint_tuples = []
            for kp in keypoints:
                # (octave_idx, y, x, s)
                keypoint_tuples.append((kp.octave_idx, kp.y, kp.x, kp.s))
            keypoints = self.compute_oriented_keypoints(keypoint_tuples, gaussian_octaves, grad_x_octaves, grad_y_octaves)
        feature_vectors = []
        # (octave_idx, y, x, s, orientation)
        for kp in keypoints:
            octave_idx = kp.octave_idx
            y = kp.y
            x = kp.x
            s = kp.s

            # scale = 2 ** (octave_idx - 1)
            y = int(np.round(y))
            x = int(np.round(x))
            s = int(np.round(s))

            grad_x_L = grad_x_octaves[octave_idx][s]
            grad_y_L = grad_y_octaves[octave_idx][s]
            feature_vectors.append(self.compute_feature_vector(kp, grad_x_L, grad_y_L))
        X = np.array(feature_vectors)

        return X, keypoints
    
    def compute_feature_vector(self, kp, grad_x_L, grad_y_L):
        theta = np.deg2rad(kp.orientation)
        # grid offsets
        offsets = np.linspace(-19, +19, 39)
        u, v = np.meshgrid(offsets, offsets) # (39,39) * 2

        R = np.array([[ np.cos(theta), -np.sin(theta)],
                      [ np.sin(theta),  np.cos(theta)]])
        uv = np.stack([u.ravel(), v.ravel()], axis=1) # (1521,2)
        rot_uv = uv.dot(R.T).reshape(39, 39, 2) # (39,39,2)
        y = kp.y
        x = kp.x
        map_x = (rot_uv[...,0] + x).astype(np.float32)        # (39,39)
        map_y = (rot_uv[...,1] + y).astype(np.float32)        # (39,39)

        sampled_grad_x_L = cv2.remap(grad_x_L, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        sampled_grad_y_L = cv2.remap(grad_y_L, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        sampled_gradients = np.stack([sampled_grad_x_L.ravel(), sampled_grad_y_L.ravel()], axis=1)
        rot_sampled_gradients = sampled_gradients.dot(R)
        feature_vector = np.concatenate([rot_sampled_gradients[:,0], rot_sampled_gradients[:,1]])
        
        feature_vector /= np.linalg.norm(feature_vector)
        return feature_vector
    
    def compute_descriptors(self, feature_vecs1, feature_vecs2):
        X = np.concatenate([feature_vecs1, feature_vecs2])

        mean = X.mean(axis=0)

        X_c = X - mean
        feature_vecs1_c = feature_vecs1 - mean
        feature_vecs2_c = feature_vecs2 - mean

        # C = X_c.T.dot(X_c) / X.shape[0]

        # eigen_vals, eigen_vecs = np.linalg.eigh(C)

        # idx = np.argsort(eigen_vals)[::-1]
        # eigen_vals = eigen_vals[idx]
        # eigen_vecs = eigen_vecs[:, idx]
        
        # components = eigen_vecs[:, :self.n] 

        # descriptors = X_c.dot(components)

        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        components = Vt[:self.n].T
        # for j in range(20):
        #     # if the dot‐product is negative, flip the SVD vector
        #     if np.dot(components[:, j], svdcomponents[:, j]) < 0:
        #         svdcomponents[:, j] *= -1
        descriptors1 = feature_vecs1_c.dot(components)
        descriptors2 = feature_vecs2_c.dot(components)
        return descriptors1, descriptors2
        # print(np.sum(np.sum(np.sqrt((descriptors - descriptors2) ** 2), axis=1)))
        # print(Vt.dot(Vt.T))