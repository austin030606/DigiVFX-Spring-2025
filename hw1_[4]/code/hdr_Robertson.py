# %%
from PIL import Image
import numpy as np
import os
import rawpy
import imageio.v3 as iio
import random
from MTB import MTB_color
import cv2

# %%
def align_images_rgb(images, offsets):
    aligned = [images[0]]
    for i, (dx, dy) in enumerate(offsets):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(images[i + 1], M, (images[i + 1].shape[1], images[i + 1].shape[0]))
        aligned.append(shifted)
    return np.stack(aligned)

def load_rgb_images(path):
    image_paths = [
        os.path.join(path, f)
        for f in os.listdir(path) 
        if f.endswith('.JPG')
    ]
    exposures = []
    images = []
    for p in sorted(image_paths):
        img = Image.open(p).convert('RGB')
        img_np = np.array(img).astype(np.float32)
        images.append(img_np)
        # If the filename is "0.1.tiff", then exposure_time = 0.1
        filename = os.path.splitext(os.path.basename(p))[0]
        exposure_time = 1/float(filename)
        exposures.append(exposure_time)
    images = np.stack(images, axis=0)  # shape (P, H, W, 3)
    exposures = np.array(exposures)    # shape (P,)
    return images, exposures


# %%
def robertson_calibration_single_channel(
    images, 
    exposure_times, 
    max_iter=20, 
    sampling=1000, 
    w_fun="debevec", 
    clamp_min=0.01, 
    clamp_max=0.99,
    random_seed=42
):
    images = images.astype(np.float64)
    if images.max() > 1.0:
        images /= 255.0

    N, H, W = images.shape
    
    def weight(z):
        # z in discrete [0..255].
        if w_fun == "uniform":
            return 1.0
        elif w_fun == "linear":
            return min(z, 255 - z)  
        elif w_fun == "debevec":
            return z * (255 - z)    
        else:
            return z * (255 - z)
    
    # 1) Sample valid pixels
    np.random.seed(random_seed)
    sample_coords = []
    tries = 0
    max_tries = sampling * 10
    while len(sample_coords) < sampling and tries < max_tries:
        r = np.random.randint(0, H)
        c = np.random.randint(0, W)
        pixel_vals = images[:, r, c]  # shape (N,)
        # only accept if all frames are within clamp_min..clamp_max
        if (pixel_vals >= clamp_min).all() and (pixel_vals <= clamp_max).all():
            sample_coords.append((r, c))
        tries += 1

    if len(sample_coords) == 0:
        raise ValueError("No valid samples found. Adjust clamp_min/clamp_max or sampling.")
    
    M = len(sample_coords)
    
    # 2) Initialize g
    g = np.zeros(256, dtype=np.float64)
    for z_idx in range(256):
        # a simple guess: g(z) = log( (z+0.5)/255 ), or linear
        val = (z_idx + 0.5)/255.0
        g[z_idx] = np.log(val + 1e-6)  # avoid log(0)

    # 3) Initialize logE for each sample
    # pick middle exposure as reference
    mid_idx = N // 2
    logE = np.zeros(M, dtype=np.float64)
    for j, (r, c) in enumerate(sample_coords):
        I = images[mid_idx, r, c]
        z_idx = int(round(I * 255))
        logE[j] = g[z_idx] - np.log(exposure_times[mid_idx])

    # Precompute weight lookup
    w_lookup = np.array([weight(z) for z in range(256)], dtype=np.float64)

    # 4) Iteration 
    for it in range(max_iter):
        # Step A: Update logE
        for j, (r, c) in enumerate(sample_coords):
            num = 0.0
            den = 0.0
            for i in range(N):
                I = images[i, r, c]
                z_idx = int(round(I * 255))
                w_z = w_lookup[z_idx]
                if w_z > 0:
                    num += w_z * (g[z_idx] - np.log(exposure_times[i]))
                    den += w_z
            if den > 0:
                logE[j] = num / den
        
        # Step B: Update g
        sum_num = np.zeros(256, dtype=np.float64)
        sum_den = np.zeros(256, dtype=np.float64)
        
        for i in range(N):
            log_t = np.log(exposure_times[i])
            for j, (r, c) in enumerate(sample_coords):
                I = images[i, r, c]
                z_idx = int(round(I * 255))
                w_z = w_lookup[z_idx]
                if w_z > 0:
                    sum_num[z_idx] += w_z * (logE[j] + log_t)
                    sum_den[z_idx] += w_z
        
        for z_idx in range(256):
            if sum_den[z_idx] > 0:
                g[z_idx] = sum_num[z_idx] / sum_den[z_idx]
        
        # Step C: Offset so the average is 0
        valid_mask = w_lookup > 0
        avg_g = np.mean(g[valid_mask])
        g -= avg_g

    return g, logE, sample_coords

def reconstruct_hdr_robertson_single_channel(
    images, 
    exposure_times, 
    g, 
    w_fun="debevec", 
    logE_clip=(-20,20)
):
    images = images.astype(np.float64)
    if images.max() > 1.0:
        images /= 255.0

    N, H, W = images.shape
    
    def weight(z):
        if w_fun == "uniform":
            return 1.0
        elif w_fun == "linear":
            return min(z, 255 - z)
        elif w_fun == "debevec":
            return z * (255 - z)
        else:
            return z * (255 - z)

    w_lookup = np.array([weight(z) for z in range(256)], dtype=np.float64)
    
    hdr = np.zeros((H, W), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)
    
    for i in range(N):
        log_t = np.log(exposure_times[i])
        # discrete intensity indices
        Z_idx = np.round(images[i]*255).astype(np.int32)
        
        g_map = g[Z_idx]
        w_map = w_lookup[Z_idx]
        
        # logE = g[z] - log_t
        logE_map = g_map - log_t
        # clamp for safety
        logE_map_clamped = np.clip(logE_map, logE_clip[0], logE_clip[1])
        E_map = np.exp(logE_map_clamped)
        
        hdr += w_map * E_map
        wsum += w_map
    
    valid = wsum > 0
    out = np.zeros_like(hdr)
    out[valid] = hdr[valid] / wsum[valid]
    return out

def robertson_calibration_color(
    images_color, 
    exposure_times, 
    max_iter=20, 
    sampling=1000, 
    w_fun="debevec", 
    clamp_min=0.01, 
    clamp_max=0.99,
    random_seed=42
):
    # Split into channels
    # shape => (N, H, W)
    images_r = images_color[..., 0]
    images_g = images_color[..., 1]
    images_b = images_color[..., 2]

    # Calibrate each channel
    g_r, logE_r, coords_r = robertson_calibration_single_channel(
        images_r, 
        exposure_times,
        max_iter=max_iter, 
        sampling=sampling, 
        w_fun=w_fun, 
        clamp_min=clamp_min, 
        clamp_max=clamp_max,
        random_seed=random_seed
    )

    g_g, logE_g, coords_g = robertson_calibration_single_channel(
        images_g, 
        exposure_times,
        max_iter=max_iter, 
        sampling=sampling, 
        w_fun=w_fun, 
        clamp_min=clamp_min, 
        clamp_max=clamp_max,
        random_seed=random_seed+1  # new seed for variety
    )

    g_b, logE_b, coords_b = robertson_calibration_single_channel(
        images_b, 
        exposure_times,
        max_iter=max_iter, 
        sampling=sampling, 
        w_fun=w_fun, 
        clamp_min=clamp_min, 
        clamp_max=clamp_max,
        random_seed=random_seed+2
    )

    return (g_r, g_g, g_b)

def reconstruct_hdr_robertson_color(
    images_color, 
    exposure_times, 
    g_r, g_g, g_b, 
    w_fun="debevec", 
    logE_clip=(-20, 20)
):
    images_r = images_color[..., 0]
    images_g = images_color[..., 1]
    images_b = images_color[..., 2]

    hdr_r = reconstruct_hdr_robertson_single_channel(
        images_r, 
        exposure_times, 
        g_r, 
        w_fun=w_fun, 
        logE_clip=logE_clip
    )
    hdr_g = reconstruct_hdr_robertson_single_channel(
        images_g, 
        exposure_times, 
        g_g, 
        w_fun=w_fun, 
        logE_clip=logE_clip
    )
    hdr_b = reconstruct_hdr_robertson_single_channel(
        images_b, 
        exposure_times, 
        g_b, 
        w_fun=w_fun, 
        logE_clip=logE_clip
    )

    # Stack back into color
    hdr_color = np.stack([hdr_r, hdr_g, hdr_b], axis=-1)
    return hdr_color



def tone_map(hdr_image, gamma=2.2, exposure=1e-7):
    img_tm = 1.0 - np.exp(-hdr_image * exposure)
    img_tm = np.clip(img_tm, 0, 1)
    img_tm = img_tm ** (1 / gamma)
    return (img_tm * 255).astype(np.uint8)


def run_robertson(input_path, output_path):
    print("Running Robertson method")
    os.makedirs(output_path, exist_ok=True)
    images, exposures = load_rgb_images(input_path)
    g_r, g_g, g_b = robertson_calibration_color(
        images, 
        exposures, 
        max_iter=30, 
        sampling=10000, 
        w_fun="debevec",
        clamp_min=0.01, 
        clamp_max=0.99,
        random_seed=200
    )
    hdr_color = reconstruct_hdr_robertson_color(
        images, 
        exposures, 
        g_r, g_g, g_b, 
        w_fun="debevec",
        logE_clip=(-20, 20)
    )
    tone_mapped = tone_map(hdr_color, gamma=2, exposure=1e-3)
    iio.imwrite(os.path.join(output_path, "output_robertson.hdr"), hdr_color.astype(np.float32))
    iio.imwrite(os.path.join(output_path, "tone_mapped_preview.jpg"), tone_mapped)
    return