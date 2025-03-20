import argparse
import os
import cv2
import numpy as np
from tone_map import *


ROOT = os.path.abspath('.') + "/"

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdr_image', type=str, default='../data/memorial.hdr', help='path to the input .hdr file')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    filename = ROOT + opt.hdr_image
    
    hdr_im = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)

    tonemap = ToneMapReinhard()
    res_Reinhard = tonemap.process(hdr_im)
    # # Tonemap HDR image
    # tonemap1 = cv2.createTonemap(gamma=2.2)
    # res_debevec = tonemap1.process(hdr_im.copy())
    # tonemap2 = cv2.createTonemap(gamma=1.3)
    # res_robertson = tonemap2.process(hdr_im.copy())

    # # Convert datatype to 8-bit and save
    # res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
    # res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
    # cv2.imwrite("ldr_debevec.jpg", res_debevec_8bit)
    # cv2.imwrite("ldr_robertson.jpg", res_robertson_8bit)

