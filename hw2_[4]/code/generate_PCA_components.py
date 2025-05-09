import glob
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from feature_detector import PCA_SIFT
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_directory", type=str, default="../data/", help="path to the input jpg images")
    parser.add_argument("--number_of_images", type=int, default=5, help="maximum number of images to compute")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()

    # image_paths = sorted(Path("../data/grail/").glob("*.jpg"))
    image_paths = sorted(Path(opt.image_directory).glob("*.jpg"))
    if len(image_paths) > opt.number_of_images:
        image_paths = image_paths[:opt.number_of_images]
    imgs   = [cv2.imread(str(p)) for p in image_paths]

    sift = PCA_SIFT()

    Kps = []
    Des = []
    for im in imgs:
        kps, des = sift.detectAndCompute(im)
        Kps.append(kps)
        Des.append(des)

    sift.compute_components(Des)