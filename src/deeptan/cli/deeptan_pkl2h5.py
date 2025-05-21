import argparse
import os

from deeptan.utils.uni import convert_pickle_to_h5


def convert_pkl_to_h5():
    parser = argparse.ArgumentParser(description="Convert a pickle file to HDF5 format.")
    parser.add_argument("-i", "--input", required=True, help="Path to a pickle file or a directory containing pickle files.")
    parser.add_argument("-o", "--output", required=False, help="Path to the output HDF5 file or directory. If not provided, output will be in the same location as the input file with .h5 extension.")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing HDF5 files without prompting.")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.endswith(".pkl"):
                    input_pkl = os.path.join(root, file)
                    if args.output is None:
                        output_h5 = input_pkl.replace(".pkl", ".h5")
                    else:
                        output_h5 = os.path.join(args.output, file.replace(".pkl", ".h5"))
                    if os.path.exists(output_h5) and not args.force:
                        print(f"File already exists: {output_h5}. Skipping.")
                    else:
                        convert_pickle_to_h5(input_pkl, output_h5)
                        print(f"Converted {input_pkl} to {output_h5}.")
    else:
        if args.output is None:
            output_h5 = args.input.replace(".pkl", ".h5")
        else:
            if os.path.isdir(args.output):
                output_h5 = os.path.join(args.output, os.path.basename(args.input).replace(".pkl", ".h5"))
            else:
                output_h5 = args.output.replace(".pkl", ".h5")

        if os.path.exists(output_h5) and not args.force:
            print(f"File already exists: {output_h5}. Skipping.")
        else:
            convert_pickle_to_h5(args.input, output_h5)
            print(f"Conversion complete: {output_h5}.")
