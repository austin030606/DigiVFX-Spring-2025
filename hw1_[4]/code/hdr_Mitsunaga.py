# %%
from PIL import Image
import numpy as np
import os
import glob
import rawpy
import imageio.v3 as iio
import random
from MTB import MTB_color
import cv2


# %%
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
def mitsunaga_nayar_calibration(
        images, 
        exposure_times, 
        degree=5, 
        num_samples=1000, 
        clamp_min=0.01, 
        clamp_max=0.99, 
        random_seed=42
    ):
    # Ensure floating [0..1]
    images = images.astype(np.float64)
    if images.max() > 1.0:
        images /= 255.0

    N, H, W = images.shape
    
    np.random.seed(random_seed)

    sample_coords = []
    max_tries = num_samples * 10  # to avoid infinite loops if not enough valid
    tries = 0

    while len(sample_coords) < num_samples and tries < max_tries:
        r = np.random.randint(0, H)
        c = np.random.randint(0, W)
        # check if all N frames have intensities for this (r,c) in the valid range
        pixel_values = images[:, r, c]

        if (pixel_values >= clamp_min).all() and (pixel_values <= clamp_max).all():
            sample_coords.append((r, c))
        tries += 1

    M = len(sample_coords)
    if M == 0:
        raise ValueError("No valid samples found. Consider lowering clamp_min or raising clamp_max.")

    K = degree

    A = np.zeros((N * M, (K + 1) + M), dtype=np.float64)
    b = np.zeros(N * M, dtype=np.float64)

    # Equation: sum_{k=0..K} a_k * I^k - L_j = log(t_i)
    for j, (r, c) in enumerate(sample_coords):
        for i in range(N):
            Z_ij = images[i, r, c]  
            row_idx = i + j * N

            # Fill polynomial part for a0..aK
            for k in range(K + 1):
                A[row_idx, k] = Z_ij ** k

            # logE_j index is (K+1)+j
            A[row_idx, (K+1) + j] = -1.0

            b[row_idx] = np.log(exposure_times[i])

    # Solve in least squares sense
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Extract polynomial coeffs: a0..aK
    a = x[:(K+1)]
    logE_samples = x[(K+1):]

    return a, logE_samples, sample_coords



def apply_response_polynomial(Z, a):

    result = np.zeros_like(Z, dtype=np.float64)
    for k, coeff in enumerate(a):
        result += coeff * (Z ** k)
    return result

def reconstruct_hdr_color(
        images, 
        exposure_times, 
        coeffs_rgb, 
        weight_fn='linear', 
        logE_clip=(-20, 20)
    ):

    images = images.astype(np.float64)
    if images.max() > 1.0:
        images /= 255.0

    N, H, W, C = images.shape
    hdr = np.zeros((H, W, C), dtype=np.float64)
    weight_sum = np.zeros((H, W, C), dtype=np.float64)

    for i in range(N):
        # loop over each channel separately
        for ch in range(C):
            Z_ch = images[i, :, :, ch]
            # polynomial => log(E) + log(t_i)
            f_Z = apply_response_polynomial(Z_ch, coeffs_rgb[ch])
            logE_ch = f_Z - np.log(exposure_times[i])

            logE_ch_clamped = np.clip(logE_ch, logE_clip[0], logE_clip[1])
            E_ch = np.exp(logE_ch_clamped)

            if weight_fn == 'linear':
                # clamp intensities so we don't rely on pure blacks/whites
                w_ch = np.clip(Z_ch, 0.01, 0.99)
            elif weight_fn == 'debevec':
                # w(Z) = Z*(1 - Z) in [0..1], ignoring extremes
                w_ch = Z_ch * (1.0 - Z_ch)
            else:
                # just use linear
                w_ch = np.clip(Z_ch, 0.01, 0.99)

            hdr[..., ch] += w_ch * E_ch
            weight_sum[..., ch] += w_ch

    valid = weight_sum > 0
    hdr_out = np.zeros_like(hdr, dtype=np.float64)
    hdr_out[valid] = hdr[valid] / weight_sum[valid]

    return hdr_out


def tone_map(hdr_image, gamma=2.2, exposure=1e-7):
    img_tm = 1.0 - np.exp(-hdr_image * exposure)
    img_tm = np.clip(img_tm, 0, 1)
    img_tm = img_tm ** (1 / gamma)
    return (img_tm * 255).astype(np.uint8)

# %%
def compute_mitsunaga_nayar_error(images, exposure_times, coeffs, sample_coords):
    images = images.astype(np.float64)
    if images.max() > 1.0:
        images /= 255.0

    N, H, W = images.shape
    M = len(coeffs) - 1
    P = len(sample_coords)

    error = 0.0

    for j in range(P):
        r, c = sample_coords[j]
        for i in range(N):
            Z_ij = images[i, r, c]
            Z_ij1 = images[i, r, c]  # same pixel across images

            # Compute polynomial values
            poly_ij = sum(coeffs[m] * (Z_ij ** m) for m in range(M + 1))
            poly_ij1 = sum(coeffs[m] * (Z_ij1 ** m) for m in range(M + 1))

            # Exposure ratio R = t_j / t_{j+1} → in practice: R_{j, j+1}
            if i < N - 1:
                R = exposure_times[i] / exposure_times[i + 1]
            else:
                R = exposure_times[i - 1] / exposure_times[i]  # fallback for last frame

            term = poly_ij - R * poly_ij1
            error += term ** 2
    return error

def find_best_degree_using_mitsunaga_error(images, exposure_times, max_degree=10, num_samples=1000):
    errors = []
    coeffs_all = []

    # Extract RGB channels
    channels = [images[..., c] for c in range(3)]

    for degree in range(1, max_degree + 1):
        coeffs_rgb = []
        error_rgb = []
        sample_coords = None

        for ch in channels:
            a, _, coords = mitsunaga_nayar_calibration(
                ch, exposure_times, degree=degree, num_samples=num_samples
            )
            coeffs_rgb.append(a)
            if sample_coords is None:
                sample_coords = coords  # use same coords across channels
            error = compute_mitsunaga_nayar_error(ch, exposure_times, a, sample_coords)
            error_rgb.append(error)

        combined_error = sum(error_rgb)
        errors.append(combined_error)
        coeffs_all.append(coeffs_rgb)
        print(f"Degree {degree}: Error = {combined_error:.4f}")

    best_idx = np.argmin(errors)
    best_degree = best_idx + 1
    print(f"Best polynomial degree: {best_degree} with error {errors[best_idx]:.4f}")
    
    return best_degree, coeffs_all[best_idx], errors


def run_mitsunaga(input_path, output_path):
    print("Running Mitsunaga method")
    os.makedirs(output_path, exist_ok=True)
    images, exposures = load_rgb_images(input_path)
    sorted_indices = np.argsort(exposures)[::-1]
    images = images[sorted_indices]
    exposures = exposures[sorted_indices]
    
    # MTB
    # offsets = MTB_color(images)
    # images_aligned = align_images_rgb(images, offsets)

    # best_degree, best_coeffs_rgb, errors = find_best_degree_using_mitsunaga_error(images_aligned, exposures, max_degree=10, num_samples=1000)

        
    degree = 4
    coeffs_rgb = []

    # We'll handle R, G, B channels separately
    for ch in range(3):
        # single_channel shape => (N, H, W)
        single_channel = images[:, :, :, ch]

        a_ch, logE_samps, sample_coords = mitsunaga_nayar_calibration(
            single_channel,
            exposures,
            degree=degree,
            num_samples=2000,
            clamp_min=0.01,
            clamp_max=0.99
        )
        coeffs_rgb.append(a_ch)
    
    hdr_result = reconstruct_hdr_color(
    images,
    exposures,
    coeffs_rgb,
    weight_fn='debevec',   # or 'linear'
    logE_clip=(-20, 20)          # safe exponent clamp
    )
    
    iio.imwrite(os.path.join(output_path, "output_mitsunaga.hdr"),
            hdr_result.astype(np.float32))
    
    # for debugging
    tone_mapped = tone_map(hdr_result, gamma=1.7, exposure=0.2)
    iio.imwrite(os.path.join(output_path, "tone_mapped_preview.jpg"), tone_mapped)
    
    return
    
