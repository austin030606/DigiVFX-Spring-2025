import numpy as np
import cv2
from matplotlib import pyplot as plt

def gamma_correction(im, gamma):
    # suppose im is correct im_d hdr image
    im_gamma_corrected = ((im) ** (1 / gamma))
    return im_gamma_corrected


class ToneMap:
    def __init__(self):
        pass

    def process(self, im: np.ndarray):
        raise NotImplementedError()

class ToneMapReinhard(ToneMap):
    def __init__(
            self, 
            luminance_coefs: np.ndarray = None, 
            delta = 0.00001, 
            a = 0.18, 
            L_white = None, 
            map_type = "global", 
            gamma = None,
            alphas = np.array([1.0/(2*(2**(1/2))), 1.6/(2*(2**(1/2)))]),
            scales = np.arange(1,43,2),
            phi = 8.0,
            epsilon = 0.05):
        
        super().__init__()
        if luminance_coefs == None:
            luminance_coefs = np.array([0.06, 0.67, 0.27])
        
        self.luminance_coefs = luminance_coefs
        self.delta = delta
        self.a = a
        self.L_white = L_white
        self.map_type = map_type
        self.gamma = gamma
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
            V2 = []
            V = []
            for i, s in enumerate(self.scales):
                v1 = cv2.filter2D(L, -1, R1[i])
                v2 = cv2.filter2D(L, -1, R2[i])
                V1.append(v1)
                V2.append(v2)
                V.append((v1 - v2)/((((2 ** self.phi) * self.a)/(s ** 2)) + v1))

            # calculate s_max for each position
            s_m_idx = np.zeros((im.shape[0], im.shape[1]), np.uint)            
            for i in range(s_m_idx.shape[0]):
                for j in range(s_m_idx.shape[1]):
                    for idx in range(self.scales.size):
                        if np.abs(V[idx][i][j]) < self.epsilon:
                            s_m_idx[i][j] = idx
                        else:
                            break
            
            # apply transformation for each pixel
            Ld = np.zeros(L.shape)
            for i in range(Ld.shape[0]):
                for j in range(Ld.shape[1]):
                    Ld[i][j] = L[i][j] / (1 + V1[s_m_idx[i][j]][i][j])
            
        else:
            print("map type not implemented")
            raise NotImplementedError()

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

    def compute_world_luminance(self, im):
        L = np.zeros((im.shape[0], im.shape[1]))

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                # apply the conversion for each pixel
                # for example if luminance_coefs = [0.06, 0.67, 0.27]
                # then L = 0.06B + 0.67G + 0.27R
                L[i][j] = self.luminance_coefs.dot(im[i][j])
                
        return L
    
    def get_log_average_luminance_of(self, L):
        log_sum = 0.0
        
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                log_sum += np.log(self.delta + L[i][j])

        return np.exp(log_sum / (L.shape[0] * L.shape[1]))
    
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