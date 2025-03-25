from hdr_Debevec import run_debevec 
from hdr_Mitsunaga import run_mitsunaga
from hdr_Robertson import run_robertson
import argparse


def main():
    parser = argparse.ArgumentParser(description="HDR Radiance Map Generator")
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        default="../data/raw/", 
        help="Path to input directory containing RAW TIFF images"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="../output/debevec/", 
        help="Path to output directory"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="debevec",
        help="Mode for HDR generation"
    )
    args = parser.parse_args()
    
    if args.mode == "debevec":
        run_debevec(args.input, args.output)

    elif args.mode == "mitsunaga":
        run_mitsunaga(args.input, args.output)
    
    elif args.mode == "robertson":
        run_robertson(args.input, args.output)
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    
if __name__ == "__main__":
    main()