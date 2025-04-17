import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm
import glob

def extract_features(img):
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(desc1, desc2, ratio_thresh=0.75):
    tree = cKDTree(desc2)
    distances, indices = tree.query(desc1, k=2)

    matches = []
    for i, (d1, d2) in enumerate(distances):
        if d1 < ratio_thresh * d2:
            match = cv2.DMatch(_queryIdx=i, _trainIdx=indices[i][0], _distance=d1)
            matches.append(match)
    return matches
