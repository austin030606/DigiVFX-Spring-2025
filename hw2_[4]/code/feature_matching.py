import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm
import glob
from feature_detector import SIFT, HarrisCornerDetector, PCA_SIFT

def extract_features(img, detection_method = "SIFT", descriptor_method = "PCA_SIFT"):
    print(detection_method, descriptor_method)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # sift = cv2.SIFT_create()
    keypoints = None
    descriptors = None
    if detection_method == "SIFT":
        sift = SIFT()
        keypoints = sift.detect(img)
    elif detection_method == "PCA_SIFT":
        pca_sift = PCA_SIFT()
        keypoints = pca_sift.detect(img)
    elif detection_method == "Harris":
        harris = HarrisCornerDetector()
        keypoints = harris.detect(img)
    else:
        raise RuntimeError(f"keypoint detection method \"{detection_method}\" not found")
        
    if descriptor_method == "SIFT":
        sift = SIFT()
        descriptors, keypoints = sift.compute(keypoints, img)
    elif descriptor_method == "PCA_SIFT":
        pca_sift = PCA_SIFT()
        descriptors, keypoints = pca_sift.compute(keypoints, img)
    else:
        raise RuntimeError(f"descriptor method \"{descriptor_method}\" not found")
    
    return keypoints, descriptors

def match_features(desc1, desc2, ratio_thresh=0.75, descriptor_method = "SIFT"):
    if descriptor_method == "PCA_SIFT":
        # PCA_SIFT vectors needs to be projected into the same space before comparison
        pca_sift = PCA_SIFT()
        desc1, desc2 = pca_sift.compute_descriptors(desc1, desc2)
    tree = cKDTree(desc2)
    distances, indices = tree.query(desc1, k=2)

    matches = []
    for i, (d1, d2) in enumerate(distances):
        if d1 < ratio_thresh * d2:
            match = cv2.DMatch(_queryIdx=i, _trainIdx=indices[i][0], _distance=d1)
            matches.append(match)
    return matches
