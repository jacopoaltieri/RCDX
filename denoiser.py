import os
import glob
import argparse
import filters
from utils import save_tif, ImageProcessor

# Default window and level values
WINDOW = 6000
LEVEL = 29000


def main(args):
    # Command-line arguments
    mask_idx = args.mask_idx
    filter_name = args.filter
    output_type = args.output_type
    output_dir = args.output_dir
    input_path = args.input_path

    # Get the list of files
    if os.path.isdir(input_path):
        files = glob.glob(f"{input_path}/*.raw")
    else:
        files = [input_path]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Match the filter name and set the filter function and parameters
    filter_func = None
    filter_args = []
    if filter_name == "bm3d":
        filter_func = filters.bm3d_pypi
    elif filter_name == "bilateral":
        filter_func = filters.bilateral_filter
        filter_args = [3, 1050]  # sigmad, sigmar
    elif filter_name == "anisodiff":
        filter_func = filters.anisodiff
        filter_args = [15, 0.25]  # kappa, gamma

    # Process each file
    for file in files:
        print(f"Processing file: {file}")

        # Initialize the ImageProcessor
        processor = ImageProcessor(
            sequence_path=file,
            window=WINDOW,
            level=LEVEL,
            mask_idx=mask_idx,
            filter_type=filter_func,
        )

        # Load the sequence
        sequence = processor.load_sequence()

        # Estimate noise for BM3D if necessary
        if filter_func == filters.bm3d_pypi:
            estimated_noise = processor.estimate_noise()
            filter_args = [estimated_noise]
            print(f"Estimated noise: {estimated_noise}")

        # Determine suffix for filenames based on filtering
        filter_suffix = f"_{filter_name}" if filter_name else ""

        # Handle output types
        if output_type in ["normal", "both"] and filter_func:
            # Filter the original sequence if a filter is provided
            filtered_sequence = processor.apply_filter(
                sequence=sequence, filter_args=filter_args
            )
            normal_output_file = os.path.join(
                output_dir,
                os.path.basename(file).replace(".raw", f"{filter_suffix}.tif"),
            )
            save_tif(filtered_sequence, normal_output_file)

        if output_type in ["dsa", "both"]:
            # Perform DSA (filtering depends on the availability of a filter)
            if filter_func:
                filtered_dsa = processor.perform_dsa(filter_args=filter_args)
            else:
                filtered_dsa = processor.perform_dsa()  # No filtering applied
            dsa_output_file = os.path.join(
                output_dir,
                os.path.basename(file).replace(".raw", f"_dsa{filter_suffix}.tif"),
            )
            save_tif(filtered_dsa, dsa_output_file)

        print(f"Finished processing file: {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process medical image sequences with filtering and DSA."
    )
    parser.add_argument(
        "input_path", type=str, help="Path to the input sequence file or directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["normal", "dsa", "both"],
        default="both",
        help="Type of output to generate: 'normal', 'dsa', or 'both'.",
    )
    parser.add_argument(
        "--mask_idx",
        type=int,
        default=0,
        help="Index of the mask frame in the sequence.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["bm3d", "bilateral", "anisodiff"],
        default=None,
        help="Type of filter to apply: 'bm3d', 'bilateral', or 'anisodiff'. If not specified, only unfiltered DSA is saved.",
    )

    args = parser.parse_args()
    main(args)
