# Example command (Robertson + Durand):
cd code
python hdr.py -i ../data/jpg/ -o ../data/ -m robertson
python tone_mapping.py --tone_map Durand --gamma 2.05 --hdr_image ../data/output_robertson.hdr --base_contrast 4.3

# Example command 2 (Mitsunaga + Fattal):
cd code
python hdr.py -i ../data/jpg/ -o ../data/ -m mitsunaga
python tone_mapping.py --tone_map Fattal --gamma 2.0 --hdr_image ../data/output_robertson.hdr --beta 0.95 --maxiter 10000 --saturation 1.3 --bc 5


# Example command 3 (Debevec + Reinhard):
cd code
python run_tiff.py
python hdr.py -i ../data/raw/ -o ../data/ -m debevec
python tone_mapping.py --tone_map Reinhard --gamma 2 --hdr_image ../data/output_robertson.hdr --tone_map_type local --key_value 0.3 --phi 8.0 --threshold 0.05 --scale 26

# Run HDR
## Debevec
### generate tiff first
python run_tiff.py
python hdr.py 

## Mitsunaga
python hdr.py -i ../data/jpg/ -o ../output/mitsunaga/ -m mitsunaga

## Robertson
python hdr.py -i ../data/jpg/ -o ../output/robertson/ -m robertson


# Run Tone Mapping
usage:
python tone_mapping.py [-h] [--hdr_image HDR_IMAGE] [--output_filename_postfix OUTPUT_FILENAME_POSTFIX]
                       [--tone_map TONE_MAP] [--gamma GAMMA] [--tone_map_type TONE_MAP_TYPE]
                       [--key_value KEY_VALUE] [--phi PHI] [--threshold THRESHOLD] [--scale SCALE]
                       [--base_contrast BASE_CONTRAST] [--limit_runtime LIMIT_RUNTIME] [--beta BETA]
                       [--maxiter MAXITER] [--saturation SATURATION] [--bc BC]

options:
  -h, --help            show a help message and exit

  --hdr_image HDR_IMAGE
                        path to the input .hdr file

  --output_filename_postfix OUTPUT_FILENAME_POSTFIX
                        output filename postfix, by default the tone mapped image would be saved as <original filename>_<tone map method>.jpg in the same directory as the hdr image
                        for example, using Reinhard's global method on output.hdr, the tone mapped image would be named as output_Reinhard_global.jpg
  
  --tone_map TONE_MAP   tone map method
                        can be either "Reinhard", "Durand", or "Fattal"
  
  --gamma GAMMA         gamma correction value


for Reinhard's method
  --tone_map_type TONE_MAP_TYPE
                        tone map type for Reinhard's method
                        can be either "global" or "local"
  
  --key_value KEY_VALUE
                        key value for Reinhard's method

  --phi PHI             phi for Reinhard's local method

  --threshold THRESHOLD
                        the threshold used for scale selection for Reinhard's local method

  --scale SCALE         scale for Reinhard's local method


for Durand's method
  --base_contrast BASE_CONTRAST
                        base contrast for Durand's method
  --limit_runtime LIMIT_RUNTIME
                        whether to limit the runtime of Durand's method, type "No" to disable


for Fattal's method
  --beta BETA           beta value for Fattal's method

  --maxiter MAXITER     max iteration for solving the poisson equation in Fattal's method
  
  --saturation SATURATION
                        saturation value for Fattal's method
  
  --bc BC               boundary condition should be active for which border when solving the poisson equation in
                        Fattal's method, represented using a 4-bit number. S
                        tarting from the left, if the first bit is 1, then it's set for the top border,
                                               if the second bit is 1, the bottom border, 
                                               the third, the left, 
                                               the fourth, the right


example command:
python tone_mapping.py --tone_map Durand --gamma 2.05 --hdr_image ../data/tone_map_results/robertson/output_classroom_01.hdr --base_contrast 4.3
