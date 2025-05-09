from homography import *
import cv2
import argparse

ROOT = os.path.abspath(".") + "/"

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_directory", type=str, default="../data/", help="path to the input jpg images")
    parser.add_argument("--output_filename", type=str, default="result.png", help="output filename")
    
    parser.add_argument("--number_of_images", type=int, default=15, help="number of images to stitch starting from the first image")
    
    parser.add_argument("--method", type=str, default="cylindrical", help="stitching method, \"cylindrical\" or \"perspective\"")
    
    parser.add_argument("--detection_method", type=str, default="Harris", help="keypoint detection method, \"Harris\" or \"SIFT\"")
    parser.add_argument("--descriptor_method", type=str, default="PCA_SIFT", help="keypoint detection method, \"SIFT\" or \"PCA_SIFT\"")
    parser.add_argument("--correct_drift", type=str, default="Yes", help="whether to correct vertical drift, \"Yes\" or \"No\"")
    parser.add_argument("--bruteforce_match", type=str, default="No", help="whether to match descriptors using the bruteforce method, \"Yes\" or \"No\"")
    parser.add_argument("--use_precompute_pca", type=str, default="Yes", help="whether to use precomputed PCA components for PCA-SIFT, \"Yes\" or \"No\", files \"components.npy\" and \"mean.npy\" must be present")
    
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    IMAGE_LIST = sorted(Path(opt.image_directory).glob("*.jpg"))
    # focals = [
    #     704.916, 706.286, 705.849, 706.645, 706.587, 705.645,
    #     705.327, 704.696, 703.794, 704.325, 704.696, 703.895,
    #     704.289, 704.676, 704.847, 704.537, 705.102, 705.576
    # ]
    N = opt.number_of_images
    # focals = [1150 for i in range(N)]

    focals = [1133.1089721 , 1134.70075868, 1137.4610752 , 1140.33852558, 1143.23703668, 1143.3166758 , 1145.49707377, 1149.52413314, 1156.19249583, 1160.26955415, 1163.92244789, 1166.19638263, 1168.61223223, 1170.21171404, 1172.08615147]
    while len(focals) < N:
        focals.append(focals[-1])
    
    panorama = stitch_images(IMAGE_LIST[:N], 
                             focals[:N], 
                             method=opt.method, 
                             blending_method="linear", 
                             detection_method=opt.detection_method, 
                             descriptor_method=opt.descriptor_method, 
                             correct_vertical_drift_at_the_end=(opt.correct_drift == "Yes"), 
                             bruteforce_match=(opt.bruteforce_match == "Yes"),
                             use_precompute_pca=(opt.use_precompute_pca == "Yes"))
    cv2.imwrite(opt.output_filename, panorama)
    

if __name__ == "__main__":
    main()