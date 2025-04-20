from homography import *
import cv2


def main():
    IMAGE_LIST = sorted(Path("../data/parrington/").glob("*.jpg"))
    focals = [
        704.916, 706.286, 705.849, 706.645, 706.587, 705.645,
        705.327, 704.696, 703.794, 704.325, 704.696, 703.895,
        704.289, 704.676, 704.847, 704.537, 705.102, 705.576
    ]

    panorama = stitch_images(IMAGE_LIST, focals, blending_method="linear", detection_method="Harris", descriptor_method="PCA_SIFT")
    cv2.imwrite("panorama.jpg", panorama)
    

if __name__ == "__main__":
    main()