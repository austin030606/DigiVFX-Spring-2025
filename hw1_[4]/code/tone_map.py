import numpy as np
import cv2
from matplotlib import pyplot as plt

class ToneMap:
    def __init__(self):
        pass

    def process(self, im: np.ndarray):
        raise NotImplementedError()

# L = 0.27R + 0.67G + 0.06B
class ToneMapReinhard(ToneMap):
    def __init__(self, luminance_coefs: np.ndarray = None, delta = 0.00001, a = 0.18):
        super().__init__()
        if luminance_coefs == None:
            luminance_coefs = np.array([0.06, 0.67, 0.27])
        
        self.luminance_coefs = luminance_coefs
        self.delta = delta
        self.a = a

    def process(self, im):
        Lw = self.compute_world_luminance(im)
        Lw_bar = self.get_log_average_luminance_of(Lw)
        
        L = (self.a / Lw_bar) * Lw
        print(L.shape, self.a / Lw_bar)
        pass

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