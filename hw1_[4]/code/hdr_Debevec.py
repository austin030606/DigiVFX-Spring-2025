# %%
from PIL import Image
import numpy as np
import os
import glob
import rawpy
import imageio.v3 as iio

# %%
# load ppm and exposure values using their name from a directory
def load_images_and_exposures_from_dir(directory):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tiff')]
    return load_images_and_exposures(image_paths)

def turn_images_to_tiff(directory):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.RAF')]
    for path in image_paths:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                gamma=(1,1),             # no gamma correction
                no_auto_bright=True,     # no auto exposure
                output_bps=16,           # 16-bit output
                use_camera_wb=True       # or False, depending on your needs
            )
            iio.imwrite(path.replace('.RAF', '.tiff'), rgb)
            
def load_images_and_exposures(image_paths):
    images = []
    exposures = []
    for path in image_paths:
        img = Image.open(path)
        img_np = np.array(img).astype(np.float32)  # ensure float for calculations
        if img_np.max() <= 255: 
           img_np*= 257
        images.append(img_np)
        # extract exposure value from filename
        exposure = float(path.split('/')[-1].split('.')[0])
        exposures.append(1/exposure)
    return np.stack(images), np.array(exposures)

def weight_function(z, z_min=0, z_max=65535):
    return 1.0 - 2.0 * np.abs((z - (z_max / 2)) / z_max)

# %%
def compute_radiance_map_log(images, exposure_times):
    P, H, W, C = images.shape
    ln_radiance_map = np.zeros((H, W, C), dtype=np.float32)
    weight_sum = np.zeros((H, W, C), dtype=np.float32)

    ln_exposure_times = np.log(exposure_times)

    for j in range(P):
        Z = images[j]  
        t_j = ln_exposure_times[j]

        Z_safe = np.clip(Z, 1, None)
        ln_Z = np.log(Z_safe)
        weight = weight_function(Z_safe)

        ln_E = ln_Z - t_j

        ln_radiance_map += weight * ln_E
        weight_sum += weight

    weight_sum[weight_sum == 0] = 1.0

    ln_radiance_map /= weight_sum
    radiance_map = np.exp(ln_radiance_map)
    

    return radiance_map

def tone_map(hdr_image, gamma=2.2, exposure=1e-7):
    img_tm = 1.0 - np.exp(-hdr_image * exposure)
    img_tm = np.clip(img_tm, 0, 1)
    img_tm = img_tm ** (1 / gamma)
    return (img_tm * 255).astype(np.uint8)



def main():
    data_path = "../data/raw/"
    output_path = "../data/output/"
    
    # turn_images_to_tiff
    # turn_images_to_tiff(data_path)
    
    images, exposures = load_images_and_exposures_from_dir(data_path)
    
    radiance_map = compute_radiance_map_log(images, exposures)
    
    tone_mapped = tone_map(radiance_map, exposure=5e-7)
    iio.imwrite(output_path + "tone_mapped.jpg", tone_mapped)
    
    radiance_map /= radiance_map.max()
    iio.imwrite(output_path + "output.hdr", radiance_map.astype(np.float32))


if __name__ == "__main__":
    main()