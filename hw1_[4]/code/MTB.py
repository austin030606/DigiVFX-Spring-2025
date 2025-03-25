import os
import cv2
import numpy as np

# input: N 8-bit grayscale images, assuming N > 1
# the first image in the list is selected as the reference reference image,
#
# output: a list of N-1 (x,y) integer, 
# representing the offsets for each of the remaining images relative to the reference
# [(dx1, dy2), (dx2, dy2), ...]

# the returned dx's and dy's can be turned into a transformation matrix as follows:
# M = np.float32([
#     [1, 0, dx],
#     [0, 1, dy]
# ])


def align_images_rgb(images, offsets):
    """
    Apply affine shift to a list of full-color (H, W, 3) images using given offsets.
    Assumes images[0] is reference and not shifted.
    """
    aligned = [images[0]]
    for i, (dx, dy) in enumerate(offsets):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(images[i + 1], M, (images[i + 1].shape[1], images[i + 1].shape[0]))
        aligned.append(shifted)
    return np.stack(aligned)


def MTB_color(color_images, max_offset=500):
    """
    MTB alignment for a list of full-color (H, W, 3) images.
    Returns: list of (dx, dy) offsets for all images relative to the first one.
    """
    def rgb_to_gray(im):
        # Weighted grayscale conversion, same as your code
        b, g, r = cv2.split(im)
        return ((54.0 * r + 183.0 * g + 19.0 * b) / 256.0).astype(np.uint8)

    # Convert all RGB images to grayscale
    gray_images = [rgb_to_gray(im) for im in color_images]

    # Call the original MTB
    return MTB(gray_images, max_offset=max_offset)




# if use_exclusion_bitmaps is True and eb_threshold is 0
# the pixels with exactly the median value will not be considered
# to consider all pixels, set use_exclusion_bitmaps to False
def MTB(images, max_offset = 500, use_exclusion_bitmaps = True, eb_threshold = 4):
    pyramid_level = np.log2(max_offset).astype(np.uint)
    ref_im = images[0]
    ref_im_pyramid = construct_pyramid(ref_im, pyramid_level)
    
    offsets = []
    for i in range(1, len(images)):
        cur_im = images[i]
        cur_im_pyramid = construct_pyramid(cur_im, pyramid_level)
        offset = calculate_offset(ref_im_pyramid, cur_im_pyramid, use_exclusion_bitmaps, eb_threshold)
        offsets.append(offset)

        # M = np.float32([
        #     [1, 0, offset[0]],
        #     [0, 1, offset[1]]
        # ])
        # ref_im_bin = binarize_using_median(ref_im)
        # cur_im_bin = binarize_using_median(cur_im)
        # shifted_cur_im_bin = cv2.warpAffine(cur_im_bin, M, (cur_im_bin.shape[1], cur_im_bin.shape[0]))
        # xor_im = ref_im_bin ^ cur_im_bin
        # cv2.imshow("binary", (xor_im) * 250)
        # xor_im_shifted = ref_im_bin ^ shifted_cur_im_bin
        # cv2.imshow("binary shifted", (xor_im_shifted) * 250)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return offsets

def binarize_using_median(im):
    median = np.median(im)
    res = (im > median).astype(np.uint8)
    # cv2.imshow("binary", (res) * 250)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res

def compute_exclusion_bitmap(im, d = 4):
    median = np.median(im)
    res1 = (im > median + d).astype(np.uint8)
    res2 = (im < median - d).astype(np.uint8)
    # cv2.imshow("eb", (res1 | res2) * 250)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res1 | res2

def construct_pyramid(im, number_of_levels):
    res = [im.copy()]
    height = im.shape[0] // 2
    width = im.shape[1] // 2
    for i in range(number_of_levels):
        if (height <= 32 or width <= 32):
            break
        cur_im = cv2.resize(im, (width,height))
        res.append(cur_im)
        height //= 2
        width //= 2

    return res

def calculate_offset(ref_pyramid, im_pyramid, use_exclusion_bitmaps = True, eb_threshold = 4):
    cur_dx = 0
    cur_dy = 0

    for k in range(len(ref_pyramid) - 1, -1, -1):
        cur_dx *= 2
        cur_dy *= 2
        cur_ref = ref_pyramid[k]
        cur_im  = im_pyramid[k]
        min_diff = max(ref_pyramid[0].shape[0], ref_pyramid[0].shape[1]) ** 2 + 1
        min_diff_dx = cur_dx
        min_diff_dy = cur_dy

        cur_ref_bin = binarize_using_median(cur_ref)
        cur_ref_eb = compute_exclusion_bitmap(cur_ref, eb_threshold)
        cur_im_bin = binarize_using_median(cur_im)
        cur_im_eb = compute_exclusion_bitmap(cur_im, eb_threshold)
        # cv2.imshow("ref pass", (cur_ref_bin & cur_ref_eb) * 250)
        # cv2.imshow("cur pass", (cur_im_bin & cur_im_eb) * 250)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                M = np.float32([
                    [1, 0, cur_dx + dx],
                    [0, 1, cur_dy + dy]
                ])

                diff = 0
                if use_exclusion_bitmaps:
                    shifted_cur_im_bin = cv2.warpAffine(cur_im_bin, M, (cur_im_bin.shape[1], cur_im_bin.shape[0]))
                    shifted_cur_im_eb = cv2.warpAffine(cur_im_eb, M, (cur_im_eb.shape[1], cur_im_eb.shape[0]))
                    bin_diff = cur_ref_bin ^ shifted_cur_im_bin
                    bin_diff = bin_diff & cur_ref_eb
                    bin_diff = bin_diff & shifted_cur_im_eb
                else:
                    shifted_cur_im_bin = cv2.warpAffine(cur_im_bin, M, (cur_im_bin.shape[1], cur_im_bin.shape[0]))
                    bin_diff = cur_ref_bin ^ shifted_cur_im_bin
                    # cv2.imshow("ref pass", (bin_diff) * 250)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                diff = np.sum(bin_diff)
                if diff < min_diff:
                    min_diff = diff
                    min_diff_dx = cur_dx + dx
                    min_diff_dy = cur_dy + dy
        
        # print(cur_dx, cur_dy)
        cur_dx = min_diff_dx
        cur_dy = min_diff_dy

    # print(cur_dx, cur_dy)
    return (cur_dx, cur_dy)

if __name__ == "__main__":
    ROOT = os.path.abspath(".") + "/"
    filename1 = ROOT + "../data/StLouisArchMultExpEV-1.82.JPG"
    filename2 = ROOT + "../data/StLouisArchMultExpEV-4.72.JPG"
    filename3 = ROOT + "../data/StLouisArchMultExpEV+1.51.JPG"
    filename4 = ROOT + "../data/StLouisArchMultExpEV+4.09.JPG"
    # filename5 = ROOT + "../data/DSCF3957.tif"

    im_1 = cv2.imread(filename1)
    im_2 = cv2.imread(filename2)
    im_3 = cv2.imread(filename3)
    im_4 = cv2.imread(filename4)
    # im_5 = cv2.imread(filename5)

    blue, green, red = cv2.split(im_1)
    im_grey_1 = ((54.0 * red + 183.0 * green + 19.0 * blue) / 256.0).astype(np.uint8)
    blue, green, red = cv2.split(im_2)
    im_grey_2 = ((54.0 * red + 183.0 * green + 19.0 * blue) / 256.0).astype(np.uint8)
    blue, green, red = cv2.split(im_3)
    im_grey_3 = ((54.0 * red + 183.0 * green + 19.0 * blue) / 256.0).astype(np.uint8)
    blue, green, red = cv2.split(im_4)
    im_grey_4 = ((54.0 * red + 183.0 * green + 19.0 * blue) / 256.0).astype(np.uint8)
    # blue, green, red = cv2.split(im_5)
    # im_grey_5 = ((54.0 * red + 183.0 * green + 19.0 * blue) / 256.0).astype(np.uint8)

    # M = np.float32([
    #     [1.25, 0, 0],
    #     [0, 1.25, 0]
    # ])
    # im_grey_2 = cv2.warpAffine(im_grey_2, M, (im_grey_2.shape[1], im_grey_2.shape[0]))
    offsets = MTB([im_grey_1, im_grey_2, im_grey_3, im_grey_4], max(im_grey_2.shape[0], im_grey_2.shape[1]) // 5, use_exclusion_bitmaps=True, eb_threshold=0)
    # offsets = MTB([im_grey_1, im_grey_2], 50)
    print(offsets)
    # cv2.imwrite(filename[:-4] + ".jpg", im_grey)
    # print(np.min(im), np.max(im))
    # print(im.shape)
    # print(im.dtype)