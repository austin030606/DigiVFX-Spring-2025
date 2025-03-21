import numpy as np
import cv2
from matplotlib import pyplot as plt

def gamma_correction(im, gamma):
    # suppose im is correct im_d hdr image
    # im_scaled = np.clip(im * 255, 0, 255)
    im_gamma_corrected = ((im) ** (1 / gamma))
    return im_gamma_corrected


class ToneMap:
    def __init__(self):
        pass

    def process(self, im: np.ndarray):
        raise NotImplementedError()

# L = 0.27R + 0.67G + 0.06B
class ToneMapReinhard(ToneMap):
    def __init__(self, luminance_coefs: np.ndarray = None, delta = 0.00001, a = 0.18, L_white = None, map_type = "global", gamma = None):
        super().__init__()
        if luminance_coefs == None:
            luminance_coefs = np.array([0.06, 0.67, 0.27])
        
        self.luminance_coefs = luminance_coefs
        self.delta = delta
        self.a = a
        self.L_white = L_white
        self.map_type = map_type
        self.gamma = gamma

    def process(self, im):
        im_d = im.copy()

        Lw = self.compute_world_luminance(im)
        Lw_bar = self.get_log_average_luminance_of(Lw)
        
        L = (self.a / Lw_bar) * Lw

        Ld = None
        if self.map_type == "global":
            if self.L_white == None:
                self.L_white = np.max(L)
            Ld = (L * (1 + (L / (self.L_white ** 2)))) / (1 + L)
        elif self.map_type == "local":
            pass
        else:
            print("map type not implemented")
            raise NotImplementedError()

        Lw_3 = np.stack([Lw, Lw, Lw], axis=2)
        Ld_3 = np.stack([Ld, Ld, Ld], axis=2)
        im_d = Ld_3 * (im / Lw_3)

        # im_d_corrected = ((im_d / np.max(im_d)) ** (1 / self.gamma)) * np.max(im_d)
        if self.gamma == None:
            return im_d
        else:
            im_d_gamma_corrected = ((im_d) ** (1 / self.gamma))
            return im_d_gamma_corrected

    def compute_world_luminance(self, im):
        L = np.zeros((im.shape[0], im.shape[1]))

        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                L[i][j] = self.luminance_coefs.dot(im[i][j])
                # print(L[i][j])
                # print((self.luminance_coefs[0] * im[i][j][0])+(self.luminance_coefs[1] * im[i][j][1])+(self.luminance_coefs[2] * im[i][j][2]))
        
        # print(np.min(L), np.max(L))
        # plt.hist(L.ravel(), 256)
        # plt.show()
        return L
    
    def get_log_average_luminance_of(self, L):
        log_sum = 0.0
        
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                log_sum += np.log(self.delta + L[i][j])

        # print(f"log_sum: {log_sum}")
        # print(f"np.exp(log_sum) / (L.shape[0] * L.shape[1]): {np.exp(log_sum) / (L.shape[0] * L.shape[1])}")
        # print(f"np.exp(log_sum / (L.shape[0] * L.shape[1])): {np.exp(log_sum / (L.shape[0] * L.shape[1]))}")
        return np.exp(log_sum / (L.shape[0] * L.shape[1]))