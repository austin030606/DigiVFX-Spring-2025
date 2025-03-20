import numpy as np
import cv2

class ToneMap:
    def __init__(self):
        pass

    def process(self, im: np.ndarray):
        raise NotImplementedError()


class ToneMapReinhard(ToneMap):
    def __init__(self):
        super().__init__()

    def process(self, im):
        pass