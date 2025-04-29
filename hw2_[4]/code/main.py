from homography import *
import cv2


def main():
    IMAGE_LIST = sorted(Path("../data/resized/").glob("*.jpg"))
    focals = [
        704.916, 706.286, 705.849, 706.645, 706.587, 705.645,
        705.327, 704.696, 703.794, 704.325, 704.696, 703.895,
        704.289, 704.676, 704.847, 704.537, 705.102, 705.576
    ]

    focals = [1200 for i in range(18)]

    N = 15
    panorama = stitch_images(IMAGE_LIST[16:16+N], 
                             focals[:N], 
                             method="perspective", 
                             blending_method="linear", 
                             detection_method="Harris", 
                             descriptor_method="PCA_SIFT", 
                             correct_vertical_drift_at_the_end=True, 
                             bruteforce_match=False,
                             ransac="homography")
    cv2.imwrite("panorama bundle library.jpg", panorama)
    

if __name__ == "__main__":
    main()