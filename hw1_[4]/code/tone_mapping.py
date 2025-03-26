import argparse
import os
import cv2
import numpy as np
from tone_map import *


ROOT = os.path.abspath(".") + "/"

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdr_image", type=str, default="../data/memorial.hdr", help="path to the input .hdr file")
    parser.add_argument("--output_filename_postfix", type=str, default=None, help="output filename postfix")
    
    parser.add_argument("--tone_map", type=str, default="Reinhard", help="tone map method")
    parser.add_argument("--gamma", type=float, default=None, help="gamma correction value")
    
    parser.add_argument("--tone_map_type", type=str, default="local", help="tone map type for Reinhard's method")
    parser.add_argument("--key_value", type=float, default=0.18, help="key value for Reinhard's method")
    parser.add_argument("--phi", type=float, default=8.0, help="phi for Reinhard's local method")
    parser.add_argument("--threshold", type=float, default=0.05, help="the threshold used for scale selection for Reinhard's local method")
    parser.add_argument("--scale", type=int, default=43, help="scale for Reinhard's local method")
    
    parser.add_argument("--base_contrast", type=float, default=None, help="base contrast for Durand's method")
    parser.add_argument("--limit_runtime", type=str, default="Yes", help="whether to limit the runtime of Durand's method, type \"No\" to disable")
    
    parser.add_argument("--beta", type=float, default=0.8, help="beta value for Fattal's method")
    parser.add_argument("--maxiter", type=int, default=10000, help="max iteration for solving the poisson equation in Fattal's method")
    parser.add_argument("--saturation", type=float, default=1.1, help="saturation value for Fattal's method")
    parser.add_argument("--bc", type=int, default=0, help="boundary condition should be active for which border when solving the poisson equation in Fattal's method, represented using a 4-bit number. Starting from the left, if the first bit is 1, then it's set for the top border, if the second bit is 1, the bottom border, the third, the left, the fourth, the right")
    
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    filename = ROOT + opt.hdr_image

    hdr_im = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    # hdr_im = hdr_im[:,:-80,:]
    if opt.tone_map == "Reinhard":
        tonemap = ToneMapReinhard(gamma=opt.gamma, map_type=opt.tone_map_type, scales=np.arange(1,opt.scale,2), a=opt.key_value, phi=opt.phi, epsilon=opt.threshold)
        res_Reinhard = tonemap.process(hdr_im.copy())
        
        if opt.gamma == None:
            res_Reinhard = gamma_correction(res_Reinhard, 2.2)
        
        res_Reinhard_corrected_8bit = np.clip(res_Reinhard*255, 0, 255).astype("uint8")

        if opt.output_filename_postfix == None:
            cv2.imwrite(filename[:-4] + "_" + opt.tone_map + "_" + opt.tone_map_type + ".jpg", res_Reinhard_corrected_8bit)
        else:
            cv2.imwrite(filename[:-4] + "_" + opt.output_filename_postfix, res_Reinhard_corrected_8bit)
    elif opt.tone_map == "Durand":
        tonemap = ToneMapDurand(gamma=opt.gamma, base_contrast=opt.base_contrast, limit_runtime=opt.limit_runtime)
        res_Durand = tonemap.process(hdr_im.copy())

        if opt.gamma == None:
            res_Durand = gamma_correction(res_Durand, 2.2)
        
        res_Durand_corrected_8bit = np.clip(res_Durand*255, 0, 255).astype("uint8")

        if opt.output_filename_postfix == None:
            cv2.imwrite(filename[:-4] + "_" + opt.tone_map + ".jpg", res_Durand_corrected_8bit)
        else:
            cv2.imwrite(filename[:-4] + "_" + opt.output_filename_postfix, res_Durand_corrected_8bit)
    elif opt.tone_map == "Fattal":
        tonemap = ToneMapFattal(gamma=opt.gamma, beta=opt.beta, maxiter=opt.maxiter, saturation=opt.saturation, boundary_condition=opt.bc)

        if hdr_im.shape[0] > hdr_im.shape[1]:
            if hdr_im.shape[0] > 2560:
                scale = 2000 / hdr_im.shape[0]
                hdr_im = cv2.resize(hdr_im, (0, 0), cv2.INTER_LINEAR, scale, scale)
        else:
            if hdr_im.shape[1] > 2560:
                scale = 2000 / hdr_im.shape[1]
                hdr_im = cv2.resize(hdr_im, (0, 0), cv2.INTER_LINEAR, scale, scale)

        res_Fattal = tonemap.process(hdr_im.copy())

        if opt.gamma == None:
            res_Fattal = gamma_correction(res_Fattal, 2.2)
        
        res_Fattal_corrected_8bit = np.clip(res_Fattal*255, 0, 255).astype("uint8")

        if opt.output_filename_postfix == None:
            cv2.imwrite(filename[:-4] + "_" + opt.tone_map + ".jpg", res_Fattal_corrected_8bit)
        else:
            cv2.imwrite(filename[:-4] + "_" + opt.output_filename_postfix, res_Fattal_corrected_8bit)

