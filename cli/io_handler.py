"""This module handles IO operations

This includes command line arguments, as well as file system interactions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import PIL
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


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


def has_k_flag(arguments: argparse.Namespace) -> bool:
    if arguments.top_k is not None:
        return True
    return False


def has_category_flag(arguments: argparse.Namespace) -> bool:
    if arguments.label_map is not None:
        return True
    return False


def load_label_map(arguments: argparse.Namespace) -> dict | None:
    label_path = arguments.label_map
    if label_path is None:
        return None
    with open(label_path, "r") as file:
        label_map = json.load(file)
    return label_map


def get_optional_flags(
    arguments: argparse.Namespace
) -> tuple[int | None, dict | None]:
    k_value = arguments.top_k
    label_map = load_label_map(arguments)
    return k_value, label_map


def load_model(model_path: Path) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects={"KerasLayer": hub.KerasLayer},
        )
    except ImportError:
        print("HDF5 loading unavailable!")
        raise
    except IOError:
        print("Invalid save file!")
        raise


def load_image(image_path: Path) -> np.ndarray:
    try:
        with Image.open(image_path) as im:
            image_array = np.asarray(im)  # ignore this warning - this is the recommended way as of Pillow 9.2.0
        return image_array
    except PIL.UnidentifiedImageError:
        print("Unable to identify/open the image!")
        raise


def load(arguments: argparse.Namespace) -> (np.ndarray, tf.keras.Model):
    return load_image(arguments.image_path), load_model(arguments.model_path)
