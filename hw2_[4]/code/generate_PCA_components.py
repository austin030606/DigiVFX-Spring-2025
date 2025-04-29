import glob
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from feature_detector import PCA_SIFT

image_paths = sorted(Path("../data/grail/").glob("*.jpg"))
imgs   = [cv2.imread(str(p)) for p in image_paths]

sift = PCA_SIFT()

Kps = []
Des = []
for im in imgs:
    kps, des = sift.detectAndCompute(im)
    Kps.append(kps)
    Des.append(des)

sift.compute_components(Des)