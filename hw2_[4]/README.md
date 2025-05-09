pip install -r requirements.txt

# Example command 1 (Bundle adjustment of all of the images, may take a long time to run; sometimes it may fail to optimize, if that happens, try stitching less images at once or try other detection descriptor combinations):
cd code
python main.py --image_directory ../data/ --output_filename result_original.png --number_of_images 15 --method perspective --detection_method Harris --descriptor_method PCA_SIFT --correct_drift Yes --bruteforce_match No --use_precompute_pca No

# Example command 2 (Cylindrical stitching of all of the images):
cd code
python main.py --image_directory ../data/ --output_filename result_original.png --number_of_images 15 --method cylindrical --detection_method Harris --descriptor_method PCA_SIFT --correct_drift Yes --bruteforce_match No --use_precompute_pca Yes

# Example command 3 (Cropping an image):
cd code
python cropping.py --image_file result_original.png --threshold 0.1

# Example command 4 (Precompute PCA components.npy and mean.npy files):
cd code
python generate_PCA_components.py --image_directory ../data/ --number_of_images 3


use the "--help" parameter to see usage info, for example "python main.py --help" gives:
usage: main.py [-h] [--image_directory IMAGE_DIRECTORY] [--output_filename OUTPUT_FILENAME] [--number_of_images NUMBER_OF_IMAGES] [--method METHOD]
               [--detection_method DETECTION_METHOD] [--descriptor_method DESCRIPTOR_METHOD] [--correct_drift CORRECT_DRIFT] [--bruteforce_match BRUTEFORCE_MATCH]
               [--use_precompute_pca USE_PRECOMPUTE_PCA]

options:
  -h, --help            show this help message and exit
  --image_directory IMAGE_DIRECTORY
                        path to the input jpg images
  --output_filename OUTPUT_FILENAME
                        output filename
  --number_of_images NUMBER_OF_IMAGES
                        number of images to stitch starting from the first image
  --method METHOD       stitching method, "cylindrical" or "perspective"
  --detection_method DETECTION_METHOD
                        keypoint detection method, "Harris" or "SIFT"
  --descriptor_method DESCRIPTOR_METHOD
                        keypoint detection method, "SIFT" or "PCA_SIFT"
  --correct_drift CORRECT_DRIFT
                        whether to correct vertical drift, "Yes" or "No"
  --bruteforce_match BRUTEFORCE_MATCH
                        whether to match descriptors using the bruteforce method, "Yes" or "No"
  --use_precompute_pca USE_PRECOMPUTE_PCA
                        whether to use precomputed PCA components for PCA-SIFT, "Yes" or "No", files "components.npy" and "mean.npy" must be present