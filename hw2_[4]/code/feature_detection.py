import argparse
import os
import cv2
import numpy as np
from feature_detector import SIFT, HarrisCornerDetector


ROOT = os.path.abspath(".") + "/"

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../data/parrington/prtn00.jpg", help="path to the input image file")
    # parser.add_argument("--output_filename_postfix", type=str, default=None, help="output filename postfix")
    
    # parser.add_argument("--tone_map", type=str, default="Reinhard", help="tone map method")
    # parser.add_argument("--gamma", type=float, default=None, help="gamma correction value")
    
    # parser.add_argument("--tone_map_type", type=str, default="local", help="tone map type for Reinhard's method")
    # parser.add_argument("--key_value", type=float, default=0.18, help="key value for Reinhard's method")
    # parser.add_argument("--phi", type=float, default=8.0, help="phi for Reinhard's local method")
    # parser.add_argument("--threshold", type=float, default=0.05, help="the threshold used for scale selection for Reinhard's local method")
    # parser.add_argument("--scale", type=int, default=43, help="scale for Reinhard's local method")
    
    # parser.add_argument("--base_contrast", type=float, default=None, help="base contrast for Durand's method")
    # parser.add_argument("--limit_runtime", type=str, default="Yes", help="whether to limit the runtime of Durand's method, type \"No\" to disable")
    
    # parser.add_argument("--beta", type=float, default=0.8, help="beta value for Fattal's method")
    # parser.add_argument("--maxiter", type=int, default=10000, help="max iteration for solving the poisson equation in Fattal's method")
    # parser.add_argument("--saturation", type=float, default=1.1, help="saturation value for Fattal's method")
    # parser.add_argument("--bc", type=int, default=0, help="boundary condition should be active for which border when solving the poisson equation in Fattal's method, represented using a 4-bit number. Starting from the left, if the first bit is 1, then it's set for the top border, if the second bit is 1, the bottom border, the third, the left, the fourth, the right")
    
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    filename = ROOT + opt.image

    # im = cv2.imread(filename)
    # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
 
    # sift = cv2.SIFT_create()
    # kps, des = sift.detectAndCompute(im_gray,None)
    # print(len(des[0]))
    # # # img=cv2.drawKeypoints(im_gray,kp,im)
    # # # cv2.imshow(f"keypoints", img)
    # # # cv2.waitKey(0)
    # # # cv2.destroyAllWindows()
    # # # 2. make a copy to draw on
    # img_with_arrows = cv2.cvtColor(im_gray.copy(), cv2.COLOR_GRAY2BGR)

    # # 3. for each keypoint, draw an arrow
    # arrow_length = 15  # tweak to taste
    # for kp in kps:
    #     x, y = kp.pt
    #     x, y = int(round(x)), int(round(y))
    #     θ = kp.angle * np.pi / 180.0
    #     cv2.circle(img_with_arrows, (x, y), 2, (0, 255, 0), 1)
    # #     L    = kp.size      # arrow length = the diameter of the keypoint region
    # #     dx   = int(round(L * np.cos(θ)))
    # #     dy   = int(round(L * np.sin(θ)))
    # #     start = (int(round(x)), int(round(y)))
    # #     tip   = (start[0] + dx, start[1] + dy)

    # #     # draw a green arrow; tipLength=0.3 means head is 30% of arrow length
    # #     cv2.arrowedLine(img_with_arrows, start, tip, (0,255,0), 1, tipLength=0.3)

    # # # 4. display
    # cv2.imshow("oriented keypoints", img_with_arrows)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # sift = SIFT()

    # kps, des = sift.detectAndCompute(im_gray)
    # print(len(kps), len(des), len(des[0]))
    # print(des)

    harris = HarrisCornerDetector()

    kps = harris.detect(im_gray)