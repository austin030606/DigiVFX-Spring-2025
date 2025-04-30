import cv2
import numpy as np
from feature_detector import SIFT, PCA_SIFT, HarrisCornerDetector

img1 = cv2.imread('../data/parrington/prtn00.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../data/parrington/prtn00.jpg', cv2.IMREAD_GRAYSCALE)

# H = np.loadtxt('../data/H1to2p')

my_sift = SIFT()

# my_kps1, my_desc1 = my_sift.detectAndCompute(img2)
kps1, desc1 = my_sift.detectAndCompute(img1)
# kps2, desc2 = my_sift.detectAndCompute(img2)
# harris = HarrisCornerDetector()
# kps1 = harris.detect(img1)
# kps2 = harris.detect(img2)

# desc1, kps1 = my_sift.compute(kps1, img1)
# desc2, kps2 = my_sift.compute(kps2, img2)

# desc1, desc2 = my_sift.project_descriptors(desc1, desc2, 'components.npy', 'mean.npy')

# desc1 = np.array(desc1).astype(np.float32)
# desc2 = np.array(desc2).astype(np.float32)

# sift = cv2.SIFT_create()
# kps1, desc1 = sift.detectAndCompute(img2,None)
# # kps2, desc2 = sift.detectAndCompute(img2,None)
# # # print(desc1)
# # # print(desc1.shape)


my_img_with_arrows = cv2.cvtColor(img2.copy(), cv2.COLOR_GRAY2BGR)

# 3. for each keypoint, draw an arrow
arrow_length = 15  # tweak to taste
for kp in kps1:
    octave_idx = kp.octave_idx
    y = kp.y
    x = kp.x
    coord_scale = 2 ** (octave_idx - 1)
    y = int(np.round(y * coord_scale))
    x = int(np.round(x * coord_scale))
    # cv2.circle(my_img_with_arrows, (x, y), 2, (0, 255, 0), 2)

    angle = kp.orientation
    end_x = int(x + 5 * coord_scale * np.cos(np.deg2rad(angle)))
    end_y = int(y + 5 * coord_scale * np.sin(np.deg2rad(angle)))
    cv2.arrowedLine(my_img_with_arrows, (x, y), (end_x, end_y), (0, 255, 0), 1, tipLength=0.3)
# # 4. display
cv2.imshow("my oriented keypoints", my_img_with_arrows)
cv2.imwrite("parrington 0 SIFT 31600410.jpg", my_img_with_arrows)

# # img_with_arrows = cv2.cvtColor(img2.copy(), cv2.COLOR_GRAY2BGR)

# # # 3. for each keypoint, draw an arrow
# # arrow_length = 15  # tweak to taste
# # for kp in kps2:
# #     x, y = kp.pt
# #     x, y = int(round(x)), int(round(y))
# #     θ = kp.angle * np.pi / 180.0
# #     # cv2.circle(img_with_arrows, (x, y), 2, (0, 255, 0), 1)
# #     L    = kp.size      # arrow length = the diameter of the keypoint region
# #     dx   = int(round(L * np.cos(θ)))
# #     dy   = int(round(L * np.sin(θ)))
# #     start = (int(round(x)), int(round(y)))
# #     tip   = (start[0] + dx, start[1] + dy)

# #     # draw a green arrow; tipLength=0.3 means head is 30% of arrow length
# #     cv2.arrowedLine(img_with_arrows, start, tip, (0,255,0), 1, tipLength=0.3)

# # 4. display
# # cv2.imshow("oriented keypoints", img_with_arrows)
cv2.waitKey(0)
cv2.destroyAllWindows()


exit()
bf = cv2.BFMatcher(cv2.NORM_L2)
# k=2 for Lowe’s ratio test
matches = bf.knnMatch(desc1, desc2, k=2)

good = []
ratio_thresh = 0.75
for m,n in matches:
    if m.distance < ratio_thresh * n.distance:
        good.append([m])

# def is_inlier(kp1, kp2, H, eps=2.0):
#     # homogeneous coords
#     x1 = np.array([kp1.pt[0], kp1.pt[1], 1.0])
#     x2_proj = H.dot(x1)
#     x2_proj /= x2_proj[2]
#     # Euclidean distance to actual kp2
#     return np.linalg.norm(x2_proj[:2] - kp2.pt) < eps

# inliers = [m for m in good
#            if is_inlier(kps1[m.queryIdx], kps2[m.trainIdx], H)]

# print(f"precision: {len(inliers) / len(good)}")
# print(f"{len(inliers)} / {len(good)}")
# exit()

cvkps1 = []
cvkps2 = []
for i in range(len(kps1)):
    curkp1 = cv2.KeyPoint()
    curkp1.pt = kps1[i].pt.astype(np.float32)
    cvkps1.append(curkp1)

for i in range(len(kps2)):
    curkp2 = cv2.KeyPoint()
    curkp2.pt = kps2[i].pt.astype(np.float32)
    cvkps2.append(curkp2)
# print(cvkps1[0].pt)
# print(kps1[0].pt)
print(len(good))
img3 = cv2.drawMatchesKnn(img1,tuple(cvkps1),img2,tuple(cvkps2),good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("match", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()