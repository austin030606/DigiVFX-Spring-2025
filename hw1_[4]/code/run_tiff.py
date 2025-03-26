
import os
import rawpy
import imageio.v3 as iio
import argparse

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
            

def main():
    parser = argparse.ArgumentParser(description='Convert raw images to tiff')
    parser.add_argument("--input","-i", type=str, default='../data/raw/', help='Path to the directory containing raw images')
    args = parser.parse_args()

    data_path = args.input
    
    turn_images_to_tiff(data_path)

if __name__ == "__main__":
    main()