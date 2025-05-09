import cv2
import numpy as np
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str, default="result.png", help="path to the input image to crop")
    
    parser.add_argument("--threshold", type=float, default=0.1, help="threshold for percentage of black pixels for each row and column")
    
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    # # 1. load and build mask
    img = cv2.imread(opt.image_file)
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray == 0).astype(np.int32)

    col_frac = mask.sum(axis=0) / h
    row_frac = mask.sum(axis=1) / w

    threshold = opt.threshold
    valid_cols = np.where(col_frac <= threshold * 1)[0]
    valid_rows = np.where(row_frac <= threshold * 0.5)[0]
    # print(row_frac)
    # exit()
    # # 3. crop
    if valid_rows.size > 0:
        top, bottom = valid_rows[0], valid_rows[-1]
    else:
        top, bottom = 0, h-1
    if valid_cols.size > 0:
        left, right = valid_cols[0], valid_cols[-1]
    else:
        left, right = 0, w-1
    cropped = img[top:bottom, left:right]
    # # cropped = img
    cv2.imwrite("cropped "+opt.image_file, cropped)
    # mid = img.shape[1] // 2
    # cv2.imwrite('panorama bundle parrington cropped left.jpg', img[:, :mid])
    # cv2.imwrite('panorama bundle parrington cropped right.jpg', img[:, mid:])