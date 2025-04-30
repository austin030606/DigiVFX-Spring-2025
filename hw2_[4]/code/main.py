from homography import *
import cv2


def main():
    IMAGE_LIST = sorted(Path("../data/resized/").glob("*.jpg"))
    focals = [
        704.916, 706.286, 705.849, 706.645, 706.587, 705.645,
        705.327, 704.696, 703.794, 704.325, 704.696, 703.895,
        704.289, 704.676, 704.847, 704.537, 705.102, 705.576
    ]

    focals = [1150 for i in range(18)]

    # focals = [1133.1089721 , 1134.70075868, 1137.4610752 , 1140.33852558, 1143.23703668, 1143.3166758 , 1145.49707377, 1149.52413314, 1156.19249583, 1160.26955415, 1163.92244789, 1166.19638263, 1168.61223223, 1170.21171404, 1172.08615147]

    N = 15
    panorama = stitch_images(IMAGE_LIST[16:16+N], 
                             focals[:N], 
                             method="perspective", 
                             blending_method="linear", 
                             detection_method="Harris", 
                             descriptor_method="PCA_SIFT", 
                             correct_vertical_drift_at_the_end=False, 
                             bruteforce_match=False)
    cv2.imwrite("panorama bundle library drift.jpg", panorama)
    

if __name__ == "__main__":
    main()