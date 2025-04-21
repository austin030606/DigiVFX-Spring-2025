import sys
import numpy as np
import cv2
import random
from pathlib import Path
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import defaultdict
import glob
from feature_matching import *
# from blending import poisson_blend
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg   # conjugate–gradient


def similarity_transform(points):
    centroid = np.mean(points, axis=0)
    
    av_dist = np.mean(np.linalg.norm(points - centroid, axis=1))
    scale = np.sqrt(2) / av_dist
    
    sim_trans = np.array([
    [scale, 0, -scale * centroid[0]],
    [0, scale, -scale * centroid[1]],
    [0, 0, 1]])
    
    normalized_points = np.dot(sim_trans, np.vstack((points.T, np.ones(len(points))))).T[:, :2]
    return normalized_points, sim_trans


def homography_matrix(points1, points2, num, norm):
    
    A= np.zeros((num * 2, 9))
    indices_range = len(points1)
    if num > indices_range:
        print("num is larger than the number of correspondences")
        return
    if(norm):
        points1, T = similarity_transform(points1)
        points2, T_prime = similarity_transform(points2)
    #random.seed(0)
    random_indices = random.sample(range(indices_range), num)
    #random_indices = list(range(10, indices_range+10))
    for i in range(num):
        u1, v1 = points1[random_indices[i]]
        u2, v2 = points2[random_indices[i]]
        A[2*i] = [0,0,0 ,-u1,-v1,-1 ,v2*u1,v2*v1,v2]
        A[2*i+1] = [u1,v1,1,0,0,0 ,-u2*u1,-u2*v1,-u2]

    U, S, V = np.linalg.svd(A)
    if(norm):
        H_hat = V[-1].reshape(3,3)
        H = np.dot(np.linalg.inv(T_prime), np.dot(H_hat, T))
    else:
        H = V[-1].reshape(3,3)
    return H, random_indices


# calculate L2 error on gd pairs
def projection(pts, H, eps=1e-8):
    # pts : (N,2)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])   # → (N,3)
    proj  = (H @ pts_h.T).T                                # → (N,3)

    w = proj[:, 2]
    valid = np.abs(w) > eps

    proj[valid, :2] /= w[valid, None]    # safe division only on valid rows
    proj[~valid, :2]  = np.nan           # mark “at infinity” as NaN (or 0)

    return proj[:, :2]


def normalise_H(H, eps=1e-8):
    if abs(H[2,2]) > eps:
        H = H / H[2,2]
    return H


def L2_error(p_t, p_t_hat):
    error = np.linalg.norm(p_t - p_t_hat, axis=1).mean()
    return error

def find_best_h(points1, points2, p_s, p_t, epoch=1000, num=4, norm=False, threshold=4.0):
    best_H = None
    best_inliers = []
    min_error = float('inf')
    num_pts = len(points1)

    for i in range(epoch):
        H, random_indices = homography_matrix(points1, points2, num, norm)
        if H is None:
            continue
        p_t_hat = projection(p_s, H)
        errors = np.linalg.norm(p_t - p_t_hat, axis=1)
        inliers = np.where(errors < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_H = H
            best_inliers = inliers
            min_error = errors[inliers].mean() if len(inliers) > 0 else float('inf')

    if len(best_inliers) >= 4:
        best_H, _ = homography_matrix(points1[best_inliers], points2[best_inliers], num=len(best_inliers), norm=norm)

    return best_H, best_inliers, min_error


def ransac_homography(src_pts, dst_pts, thresh=4.0, max_iter=1000, norm=True):
    num_pts = src_pts.shape[0]
    best_inliers = []
    best_H = None

    for _ in range(max_iter):
        H, _ = homography_matrix(src_pts, dst_pts, num=4, norm=norm)
        if H is None:
            continue

        # Project all source points using H
        projected_pts = projection(src_pts, H)

        # Compute Euclidean distance error to actual dst_pts
        errors = np.linalg.norm(dst_pts - projected_pts, axis=1)

        # Determine inliers
        inliers = np.where(errors < thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    if len(best_inliers) >= 4:
        best_H, _ = homography_matrix(src_pts[best_inliers], dst_pts[best_inliers], num=len(best_inliers), norm=norm)

    return best_H, best_inliers


def ransac_translation(pts1, pts2, thresh=2.0, max_iter=2000):
    N = pts1.shape[0]
    best_inl, best_t = [], None

    for _ in range(max_iter):
        i = np.random.randint(N)
        t = pts2[i] - pts1[i]                # candidate translation
        err = np.linalg.norm((pts1 + t) - pts2, axis=1)
        inl = np.where(err < thresh)[0]

        if len(inl) > len(best_inl):
            best_inl, best_t = inl, t

    # refine with all inliers
    if len(best_inl):
        best_t = (pts2[best_inl] - pts1[best_inl]).mean(axis=0)

    dx, dy = best_t
    H = np.array([[1, 0, dx],
                  [0, 1, dy],
                  [0, 0,  1]], dtype=np.float64)
    return H, best_inl


def cylindrical_projection(img, f):

    h, w = img.shape[:2]
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0,   1 ]])

    # build inverse‑map for every destination pixel
    y_i, x_i = np.indices((h, w))
    x        = (x_i - w/2) / f
    y        = (y_i - h/2) / f
    sinx     = np.sin(x);  cosx = np.cos(x)

    # project back onto the original image plane (z = cosx)
    x_src = f * sinx / cosx + w/2
    y_src = f * y    / cosx + h/2

    map_x = x_src.astype(np.float32)
    map_y = y_src.astype(np.float32)
    cyl   = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    mask  = (map_x > 0) & (map_x < w-1) & (map_y > 0) & (map_y < h-1)
    mask  = mask.astype(np.uint8) * 255
    return cyl, mask

def correct_vertical_drift(H_to_ref, correct_vertical_drift_at_the_end):
    # return H_to_ref
    # 1. collect horizontal / vertical translations
    dx = np.array([H[0, 2] / H[2, 2] for H in H_to_ref])
    dy = np.array([H[1, 2] / H[2, 2] for H in H_to_ref])

    # 2. least‑squares line  dy = a·dx + b
    A = np.vstack([dx, np.ones_like(dx)]).T
    a, b = np.linalg.lstsq(A, dy, rcond=None)[0]

    if correct_vertical_drift_at_the_end:
        H_global = np.array([[1.0, 0.0, 0.0],
                            [ -a, 1.0,  -b],
                            [0.0, 0.0, 1.0]])
        
        return H_to_ref, H_global
    # 3. build corrected homographies
    H_corr = []
    for H in H_to_ref:
        tx = H[0, 2] / H[2, 2]
        ty = H[1, 2] / H[2, 2]

        # amount of drift to remove for this frame
        drift = a * tx + b

        T_corr = np.array([[1, 0, 0        ],
                           [0, 1, -drift   ],
                           [0, 0, 1        ]], dtype=np.float64)

        H_corr.append(T_corr @ H)

    return H_corr, None



def warp_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Warp img1 into panorama space
    corners_img1 = np.array([[0,0], [w1,0], [w1,h1], [0,h1]], dtype=np.float32)
    warped_corners = projection(corners_img1, H)
    
    # Find bounding box for panorama
    all_corners = np.vstack((warped_corners, [[0,0], [w2,0], [w2,h2], [0,h2]]))
    [xmin, ymin] = np.floor(all_corners.min(axis=0)).astype(int)
    [xmax, ymax] = np.ceil(all_corners.max(axis=0)).astype(int)

    # Translation to handle negative coordinates
    translation = np.array([[1, 0, -xmin],
                            [0, 1, -ymin],
                            [0, 0, 1]])
    
    size = (xmax - xmin, ymax - ymin)
    result = cv2.warpPerspective(img1, translation @ H, size)

    # Paste img2 into the panorama
    result[-ymin:h2 - ymin, -xmin:w2 - xmin] = img2
    return result


def accumulate_homographies(H_pair, ref_idx):

    n = len(H_pair) + 1
    H_to_ref = [np.eye(3) for _ in range(n)]

    # left side (i < ref)  :  H_i→ref =  H_i→i+1 · H_{i+1}→ref
    for i in range(ref_idx - 1, -1, -1):
        H_to_ref[i] = H_pair[i] @ H_to_ref[i + 1]

    # right side (i > ref) :  H_i→ref =  inv(H_{i-1}→i) · H_{i-1}→ref
    for i in range(ref_idx + 1, n):
        H_to_ref[i] = np.linalg.inv(H_pair[i - 1]) @ H_to_ref[i - 1]

    # normalise so that H[2,2] = 1
    H_to_ref = [H / H[2, 2] for H in H_to_ref]
    return H_to_ref


# Bundle Adjustment
def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0,  s],
                     [ 0, 1,  0],
                     [-s, 0,  c]])


def make_jac_sparsity(tracks, N):
    m = 2 * len(tracks)
    n = N + 1
    J = lil_matrix((m, n), dtype=int)

    for k, (i, _, j, _,) in enumerate(tracks):
        r0 = 2 * k
        cols = (i, j, n-1)          # θ_i, θ_j, log f
        for c in cols:
            J[r0, c]     = 1        # x‑residual
            J[r0 + 1, c] = 1        # y‑residual
    return J.tocsr()

def bundle_adjust_yaw_f(tracks, img_wh, f_init=None, max_nfev=200):
    if not tracks:
        raise ValueError("tracks is empty")

    # number of images
    N = 1 + max(max(t[0], t[2]) for t in tracks)

    w, h  = img_wh
    cx, cy = w / 2.0, h / 2.0
    f0 = f_init if f_init is not None else 0.5 * w

    def K(f):
        return np.array([[f, 0, cx],
                         [0, f, cy],
                         [0, 0, 1]])

    def resid(p):
        theta = p[:-1]
        f     = np.exp(p[-1])
        K_f   = K(f)
        Kinv  = np.linalg.inv(K_f)

        errs = []
        for i, p_i, j, p_j in tracks:
            R_i, R_j = rot_y(theta[i]), rot_y(theta[j])
            H_ij = K_f @ (R_j @ R_i.T) @ Kinv

            q = H_ij @ np.append(p_i, 1.0)
            q /= q[2]
            errs.extend(q[:2] - p_j)
        return np.asarray(errs, np.float64)

    x0 = np.zeros(N + 1); x0[-1] = np.log(f0)

    # res = least_squares(resid, x0,
    #                     jac='2-point',      # finite‑difference Jacobian
    #                     loss='soft_l1', f_scale=2.0,
    #                     max_nfev=max_nfev)
    
    Jsp = make_jac_sparsity(tracks, N)
    res = least_squares(
        resid, x0,
        jac_sparsity=Jsp,          # <-- key line
        loss='soft_l1', f_scale=2.0,
        max_nfev=max_nfev,         # keep your existing cap
        x_scale='jac')

    theta_opt = res.x[:-1]
    f_opt     = np.exp(res.x[-1])
    return theta_opt, f_opt



def stitch_images(image_paths, f, blend_linear=True, use_similarity=False, method = "cylindrical", blending_method = "linear", detection_method = "Harris", descriptor_method = "PCA_SIFT", correct_vertical_drift_at_the_end = False):
    imgs   = [cv2.imread(str(p)) for p in image_paths]
    N      = len(imgs)  
    MAX_MATCHES = 400
    assert N >= 2, "Need at least two images"

    if method == "cylindrical":
        # 0. cylindrical projection
        cyl_imgs   = []
        cyl_masks  = []
        for i in range(N):                      # imgs = original BGR list
            cyl, m = cylindrical_projection(imgs[i], f[i])
            cyl_imgs.append(cyl)
            cyl_masks.append(m)
    else:
        cyl_imgs = imgs
    
    # 1. pair‑wise homographies (i  ->  i+1)
    H_pair = []
    tracks = []   
    for i in tqdm(range(N-1), desc="Pairwise matching"):
        kp1, des1 = extract_features(cyl_imgs[i], detection_method, descriptor_method)
        kp2, des2 = extract_features(cyl_imgs[i+1], detection_method, descriptor_method)

        matches  = match_features(des1, des2, descriptor_method=descriptor_method)
        
        for m in matches:
            p_i = np.float32(kp1[m.queryIdx].pt)
            p_j = np.float32(kp2[m.trainIdx].pt)
            tracks.append((i, p_i, i+1, p_j))
            
        if len(matches) < 4:
            raise RuntimeError(f"Not enough matches between {i} and {i+1}")

        # build point arrays for RANSAC
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # H,_ = ransac_homography(pts1, pts2)
        H, _ = ransac_translation(pts1, pts2)
        H_pair.append(H)

    # bundle
    if method == "perspective":
        img_h, img_w = cyl_imgs[0].shape[:2]     # they’re still perspective now
        theta, f_opt = bundle_adjust_yaw_f(tracks, (img_w, img_h))    
        
    # 2. pick middle image as reference
    ref_idx  = N // 2
    
    if method == "perspective":    
        K = np.array([[f_opt, 0, img_w/2],
                [0, f_opt, img_h/2],
                [0, 0, 1]], dtype=np.float64)

        R_y = [rot_y(t) for t in theta]

    # 3. accumulate H_i→ref
    if method == "cylindrical":
        H_to_ref = accumulate_homographies(H_pair, ref_idx)
        H_to_ref, H_global = correct_vertical_drift(H_to_ref, correct_vertical_drift_at_the_end)
    else:
            # H_i→ref = K · R_ref · R_iᵀ · K⁻¹
        Kinv  = np.linalg.inv(K)
        H_to_ref = []
        R_ref = R_y[ref_idx]
        for i in range(len(theta)):
            H = K @ (R_ref @ R_y[i].T) @ Kinv
            H /= H[2,2]
            H_to_ref.append(H)

    # 4. find panorama canvas bounds
    corners = []
    for i, img in enumerate(cyl_imgs):
        h,w = img.shape[:2]
        c   = np.array([[0,0],[w,0],[w,h],[0,h]], np.float32)
        c_w = projection(c, H_to_ref[i])
        corners.append(c_w)
    corners = np.vstack(corners)
    xmin,ymin = np.floor(corners.min(axis=0)).astype(int)
    xmax,ymax = np.ceil (corners.max(axis=0)).astype(int)
    tx, ty    = -xmin, -ymin           # translate to positive coords
    H_trans   = np.array([[1,0,tx],[0,1,ty],[0,0,1]])

    canvas_sz = (xmax-xmin, ymax-ymin)        # (W,H)

    # 5. warp & blend
    acc   = np.zeros((ymax-ymin, xmax-xmin, 3), np.float32)
    mask  = np.zeros((ymax-ymin, xmax-xmin, 1), np.float32)   # weight map

    panorama = cv2.warpPerspective(cyl_imgs[ref_idx], H_trans @ H_to_ref[ref_idx], canvas_sz)

    # blending
    if blending_method == "poisson":
        # first linear blend for poisson
        for i, img in enumerate(cyl_imgs):
            H      = H_trans @ H_to_ref[i]
            warped = cv2.warpPerspective(img, H, canvas_sz)

            # weight = 1 inside warped region, 0 outside
            gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            wmask  = (gray > 0).astype(np.float32)[...,None]
            

            # add one dimens
            # wmask  = mask.astype(np.float32)[..., None] 
            
            if blend_linear:
                # simple feather: distance to nearest zero pixel
                dist = cv2.distanceTransform((wmask[...,0]).astype(np.uint8), cv2.DIST_L2, 5)
                dist = dist / dist.max()            # 0~1
                wmask = wmask * dist[...,None]

            acc  += warped.astype(np.float32) * wmask
            mask += wmask


            panorama = (acc / np.maximum(mask,1e-8)).astype(np.uint8)

    # cv2.imshow("panorama", panorama)
    for i, img in enumerate(cyl_imgs):
        if False and i == ref_idx and blending_method == "poisson":   # the reference is already in place
            continue
        H      = H_trans @ H_to_ref[i]
        warped = cv2.warpPerspective(img, H, canvas_sz)

        # weight = 1 inside warped region, 0 outside
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        wmask  = (gray > 0).astype(np.float32)[...,None]
        

        # add one dimens
        # wmask  = mask.astype(np.float32)[..., None] 
        
        if blending_method =="linear":
            if blend_linear:
                # simple feather: distance to nearest zero pixel
                dist = cv2.distanceTransform((wmask[...,0]).astype(np.uint8), cv2.DIST_L2, 5)
                dist = dist / dist.max()            # 0~1
                wmask = wmask * dist[...,None]

            acc  += warped.astype(np.float32) * wmask
            mask += wmask
    

            panorama = (acc / np.maximum(mask,1e-8)).astype(np.uint8)

        elif blending_method == "poisson":
            # implement poisson blending here
            # cv2.imshow(f"{i}th image", warped)
            mask = (gray > 0).astype(np.uint8) * 255
            # l = np.where(mask[mask.shape[0] // 2] == 255)[0][0]
            # r = np.where(mask[mask.shape[0] // 2] == 255)[0][-1]
            # mask[:, :l+60] = 0
            # mask[:, r-60:] = 0
            # mask[:15] = np.zeros(mask.shape[1])
            # mask[mask.shape[0]-15:mask.shape[0]] = np.zeros(mask.shape[1])
            # cv2.imshow(f"{i}th mask", mask)
            mass_y, mass_x = np.where(mask > 0)
            cent_x = np.average(mass_x)
            cent_y = np.average(mass_y)
            # print(cent_x)
            height, width, _ = panorama.shape
            position = (int(cent_x), int(cent_y))
            panorama = cv2.seamlessClone(warped, panorama, mask, position, cv2.NORMAL_CLONE)
            # cv2.imshow(f"added {i}th image", panorama)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            pass
            # raise ValueError(f"Unknown blending method: {blending_method}")

    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    panorama = panorama[ys.min():ys.max()+1, xs.min():xs.max()+1]

    if method == "cylindrical" and correct_vertical_drift_at_the_end:
        height, width, _ = panorama.shape
        panorama = cv2.warpPerspective(
            panorama,
            H_global,
            (width, height),
            flags=cv2.INTER_LINEAR
        )
    return panorama


