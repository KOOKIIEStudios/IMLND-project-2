"""This module handles IO operations

This includes command line arguments, as well as file system interactions.
"""

import argparse
from pathlib import Path


def initialise_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="This Python script will use a provided HDF5 model to make "
                    "predictions for the flower type present in a provided image file.",
        epilog="Usage example: python predict.py ./test_images/rose.jpg "
               "1661623189.h5 --top_k 5 --category_names label_map.json",
    )
    parser.add_argument(
        "image_path",
        action="store",
        type=Path,
        help="The path to the image that you would like to predict",
        metavar="IMAGE PATH",
    )
    parser.add_argument(
        "model_path",
        action="store",
        nargs="?",
        default="model.h5",
        type=Path,
        help="The path to the model that you would like to use (default: %(default)s)",
        metavar="MODEL PATH",
    )
    parser.add_argument(
        "--top_k",
        "-t",
        "-k",
        action="store",
        type=int,
        help="The number of possibilities to show (default: only show the top result)",
        metavar="K-VALUE",
    )
    parser.add_argument(
        "--category_names",
        "-c",
        "-n",
        action="store",
        type=Path,
        help="The path to the label map, for converting numeric flower labels to names",
        metavar="LABEL MAP",
        dest="label_map",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        help="Version number, in the format YEAR.MAJOR.MINOR",
        version="%(prog)s 2022.1.0",
    )
    return parser


def are_paths_valid(arguments: argparse.Namespace) -> bool:
    if not arguments.image_path.is_file():
        print("Invalid image file path!")
        return False
    if not arguments.model_path.is_file():
        print("Invalid model file path!")
        return False
    if arguments.label_map is not None:
        if not arguments.label_map.is_file():
            print("Invalid category label file path!")
            return False
    return True
