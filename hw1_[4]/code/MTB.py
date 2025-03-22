import os
import cv2
import numpy as np

# input: N 8-bit grayscale images
# One of the N images is arbitrarily selected as the reference image,
# and the output of the algorithm is a series of N-1 (x,y) integer 
# offsets for each of the remaining images relative to this reference.
# here the first image in the list is selected as the reference
def MTB(images, max_offset = 500):
    pyramid_level = np.log2(max_offset).astype(np.uint)
    im_reference = images[0]
    im_reference_binary = binarize_using_median(im_reference)
    im_reference_binary_pyramid = construct_pyramid(im_reference_binary, pyramid_level)

    for i in range(1, len(images)):
        cur_im = images[i]
        cur_im_binary = binarize_using_median(cur_im)
        cur_im_binary_pyramid = construct_pyramid(cur_im_binary, pyramid_level)
        
        xor_im = im_reference_binary ^ cur_im_binary
        # print(np.min(xor_im), np.max(xor_im))
        cv2.imshow("binary", (xor_im) * 250)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def binarize_using_median(im):
    median = np.median(im)
    res = (im > median).astype(np.uint8)
    
    return res

def construct_pyramid(im, number_of_levels):
    res = []
    cur_im = im.copy()
    for i in range(number_of_levels):
        if (cur_im.shape[0] <= 1 or cur_im.shape[1] <= 1):
            break
        cv2.imshow("binary", (cur_im) * 250)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        res.append(cur_im)
        cur_im = cv2.resize(cur_im, (0,0), fx=0.5, fy=0.5)

    return res


if __name__ == "__main__":
    ROOT = os.path.abspath(".") + "/"
    filename1 = ROOT + "../data/DSCF3808.tif"
    filename2 = ROOT + "../data/DSCF3817.tif"

    im_1 = cv2.imread(filename1)
    im_2 = cv2.imread(filename2)

    blue, green, red = cv2.split(im_1)
    im_grey_1 = ((54.0 * red + 183.0 * green + 19.0 * blue) / 256.0).astype(np.uint8)
    blue, green, red = cv2.split(im_2)
    im_grey_2 = ((54.0 * red + 183.0 * green + 19.0 * blue) / 256.0).astype(np.uint8)

    MTB([im_grey_1, im_grey_2], max(im_grey_2.shape[0], im_grey_2.shape[1]) // 5)
    # cv2.imwrite(filename[:-4] + ".jpg", im_grey)
    # print(np.min(im), np.max(im))
    # print(im.shape)
    # print(im.dtype)