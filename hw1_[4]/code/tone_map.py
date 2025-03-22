import numpy as np
import cv2
from matplotlib import pyplot as plt
import gc

def gamma_correction(im, gamma):
    # suppose im is correct im_d hdr image
    im_gamma_corrected = ((im) ** (1 / gamma))
    return im_gamma_corrected


class ToneMap:
    def __init__(
            self,
            luminance_coefs = None,
            gamma = None):

        self.luminance_coefs = luminance_coefs
        self.gamma = gamma
        self.delta = 0.00001
        pass

    def process(self, im: np.ndarray):
        raise NotImplementedError()
    
    def compute_world_luminance(self, im):
        L = np.zeros((im.shape[0], im.shape[1]))
        B = im[:,:,0]
        G = im[:,:,1]
        R = im[:,:,2]
        # for i in range(im.shape[0]):
        #     for j in range(im.shape[1]):
        #         # apply the conversion for each pixel
        #         # for example if luminance_coefs = [0.06, 0.67, 0.27]
        #         # then L = 0.06B + 0.67G + 0.27R
        #         L[i][j] = self.luminance_coefs.dot(im[i][j])
        L = self.luminance_coefs[0] * B + self.luminance_coefs[1] * G + self.luminance_coefs[2] * R
        return L
    
    def get_log_average_luminance_of(self, L):
        log_sum = 0.0
        
        # for i in range(L.shape[0]):
        #     for j in range(L.shape[1]):
        #         log_sum += np.log(self.delta + L[i][j])
        L += self.delta
        log_sum = np.sum(np.log(L))

        return np.exp(log_sum / (L.shape[0] * L.shape[1]))

class ToneMapReinhard(ToneMap):
    def __init__(
            self, 
            luminance_coefs = np.array([0.06, 0.67, 0.27]), 
            gamma = None,
            delta = 0.00001, 
            a = 0.18, 
            L_white = None, 
            map_type = "global", 
            alphas = np.array([1.0/(2*(2**(1/2))), 1.6/(2*(2**(1/2)))]),
            scales = np.arange(1,43,2),
            phi = 8.0,
            epsilon = 0.05):
        
        super().__init__(luminance_coefs, gamma)
        
        self.delta = delta
        self.a = a
        self.L_white = L_white
        self.map_type = map_type
        self.alphas = alphas
        self.scales = scales
        self.phi = phi
        self.epsilon = epsilon

    def process(self, im):
        im_d = im.copy()

        Lw = self.compute_world_luminance(im)
        Lw_bar = self.get_log_average_luminance_of(Lw)
        
        L = (self.a / Lw_bar) * Lw

        Ld = None
        if self.map_type == "global":
            if self.L_white == None:
                self.L_white = np.max(L)

            # apply transformation for each pixel
            Ld = (L * (1 + (L / (self.L_white ** 2)))) / (1 + L)

        elif self.map_type == "local":
            R1 = self.compute_gaussian_kernels(1)
            R2 = self.compute_gaussian_kernels(2)

            # apply kernel and calculate V(x, y, s)
            V1 = []
            # V2 = []
            V = []
            for i, s in enumerate(self.scales):
                v1 = cv2.filter2D(L, -1, R1[i])
                v2 = cv2.filter2D(L, -1, R2[i])
                V1.append(v1)
                # V2.append(v2)
                V.append((v1 - v2)/((((2 ** self.phi) * self.a)/(s ** 2)) + v1))

            V1 = np.array(V1)
            # V2 = np.array(V2)
            V = np.array(V)
            # calculate s_max for each position
            s_m_idx = np.zeros((im.shape[0], im.shape[1]), np.uint)
            V1_s_m = np.zeros(V1[0].shape)
            for i in range(s_m_idx.shape[0]):
                for j in range(s_m_idx.shape[1]):
                    indices = np.where(np.abs(V[:,i,j]) > self.epsilon)[0]
                    if indices.size > 0:
                        idx = indices[0]
                        if idx > 0:
                            idx -= 1
                    else:
                        idx = self.scales.size - 1
                    # s_m_idx[i][j] = idx
                    V1_s_m[i][j] = V1[idx][i][j]
                    # for idx in range(self.scales.size):
                    #     if np.abs(V[idx][i][j]) < self.epsilon:
                    #         s_m_idx[i][j] = idx
                    #     else:
                    #         break
                print(f"{i}, {j}", end='\r')
            print("here")
            # apply transformation for each pixel
            Ld = np.zeros(L.shape)
            # for i in range(Ld.shape[0]):
            #     for j in range(Ld.shape[1]):
            #         Ld[i][j] = L[i][j] / (1 + V1[s_m_idx[i][j]][i][j])
            Ld = L / (1 + V1_s_m)
            print("done")

            del V
            del V1
            # del V2
            gc.collect()
            
        else:
            print("map type not implemented")
            raise NotImplementedError()

        # convert luminance back to RGB
        Lw_3 = np.stack([Lw, Lw, Lw], axis=2)
        Ld_3 = np.stack([Ld, Ld, Ld], axis=2)
        im_d = Ld_3 * (im / Lw_3)

        # print(np.min(im), np.max(im))
        # print(np.min(Lw), np.max(Lw))
        # print(np.min(Ld), np.max(Ld))
        # print(np.min(im_d), np.max(im_d))
        # apply gamma correction before returning if provided with gamma value
        if self.gamma == None:
            return im_d
        else:
            im_d_gamma_corrected = ((im_d) ** (1 / self.gamma))
            return im_d_gamma_corrected
    
    def compute_gaussian_kernels(self, alpha_i):
        kernels = []

        alpha = self.alphas[alpha_i - 1]
        for s in self.scales:
            kernel = np.zeros((s,s))
            for i in range(s):
                for j in range(s):
                    x = i - (s // 2)
                    y = j - (s // 2)
                    kernel[i][j] = (1 / (np.pi * ((alpha * s) ** 2))) * np.exp((-(x * x + y * y)) / ((alpha * s) ** 2))
            kernel /= np.sum(kernel)
            kernels.append(kernel)

        return kernels
    
class ToneMapDurand(ToneMap):
    def __init__(
            self, 
            luminance_coefs = np.array([1/61, 40/61, 20/61]),
            gamma = None,
            base_contrast = 4):
        super().__init__(luminance_coefs, gamma)
        self.sigma_s = None
        self.sigma_r = 0.4
        if base_contrast == None:
            base_contrast = 4
        self.base_contrast = base_contrast

    def process(self, im):
        self.sigma_s = 0.02 * max(im.shape[0], im.shape[1])
        im_d = im.copy()

        Lw = self.compute_world_luminance(im)
        Lw_log = np.log(Lw).astype(np.float32)

        kernel_size = int(self.sigma_s * 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # f = self.compute_gaussian_kernel(self.sigma_s, kernel_size)
        # base = np.zeros(Lw_log.shape)
        # for i in range(base.shape[0]):
        #     for j in range(base.shape[1]):
        #         weighted_I = 0
        #         k = 0
        #         for x in range(kernel_size):
        #             for y in range(kernel_size):
        #                 di = x - (kernel_size // 2)
        #                 dj = y - (kernel_size // 2)
        #                 if i + di >= 0 and i + di < Lw_log.shape[0] and j + dj >= 0 and j + dj < Lw_log.shape[1]: 
        #                     weighted_I += f[x][y] * self.intensity_gaussian(np.abs(Lw_log[i][j] - Lw_log[i + di][j + dj])) * Lw_log[i + di][j + dj]
        #                     k += f[x][y] * self.intensity_gaussian(np.abs(Lw_log[i][j] - Lw_log[i + di][j + dj])) 
        #         base[i][j] = weighted_I / k
        #     print(f"row: {i}", end='\r')
        base = cv2.bilateralFilter(Lw_log, kernel_size, self.sigma_r, self.sigma_s)

        detail = Lw_log - base
        scale = self.base_contrast / (np.max(base) - np.min(base))
        compressed = (base - np.max(base)) * scale
        Ld_log = compressed + detail
        Ld = np.exp(Ld_log)
        
        # convert luminance back to RGB
        Lw_3 = np.stack([Lw, Lw, Lw], axis=2)
        Ld_3 = np.stack([Ld, Ld, Ld], axis=2)
        im_d = Ld_3 * (im / Lw_3)

        # apply gamma correction before returning if provided with gamma value
        if self.gamma == None:
            return im_d
        else:
            im_d_gamma_corrected = ((im_d) ** (1 / self.gamma))
            return im_d_gamma_corrected
        
    def compute_gaussian_kernel(self, sigma, s):
        kernel = np.zeros((s,s))
        for i in range(s):
            for j in range(s):
                x = i - (s // 2)
                y = j - (s // 2)
                kernel[i][j] = np.exp((-(x * x + y * y)) / (2 * ((sigma) ** 2)))
        # kernel /= np.sum(kernel)
        return kernel
    
    def intensity_gaussian(self, d):
        return np.exp((-(d * d)) / (2 * ((self.sigma_r) ** 2)))