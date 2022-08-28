"""Introduction to Machine Learning Nanodegree Project 2

@author KOOKIIE
This script will predict flowers from a provided image.
This module is the entry point for the script, and is to be run from the
command line. This module requires a Keras model (in HDF5 format) to be
placed in the same directory. Note that a `label_map.json` file is provided.
Please do not remove this file, as it is used for mapping numeric dataset labels
to human-readable text labels (i.e. flower names).
An image path and model name are *required* command line arguments.
There is a `top_k` flag, for obtaining the probabilities
for an arbitrary number of possible results, and a `category_names` flag for
displaying human-readable text labels for results, rather than the numeric label
in the dataset.

    Typical usage example:
    python predict.py ./test_images/unknown_flower.jpg my_model.h5
    python predict.py ./test_images/orchid.jpg model.h5 --top_k 5
    python predict.py ./test_images/image_0001.jpg saved_model.h5 --category_names label_map.json
    python predict.py ./test_images/rose.jpg 1661623189.h5 --top_k 5 --category_names label_map.json
"""
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import tensorflow as tf

import format
import io_handler
import operation_mode


cli_parser: ArgumentParser
cli_arguments: Namespace
mode: int
k_value: int | None
label_map: dict[str, str] | None


def predict(image: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    return model(image, training=False).numpy().flatten()


if __name__ == "__main__":
    print("Flower prediction script started.")
    # Grab CLI arguments
    cli_parser = io_handler.initialise_parser()
    cli_arguments = cli_parser.parse_args()

    # Sanity check:
    # print(vars(cli_arguments))  # debug print
    if not io_handler.are_paths_valid(cli_arguments):
        sys.exit("This program will now terminate.")

    # Set variables:
    mode = operation_mode.get_mode(
        io_handler.has_k_flag(cli_arguments),
        io_handler.has_category_flag(cli_arguments),
    )
    image, model = io_handler.load(cli_arguments)
    k_value, label_map = io_handler.load_label_map(cli_arguments)

    # Make prediction:
    prediction = format.convert_to_dataframe(predict(image, model))
    output_text: list[str] = format.format_output(mode, prediction, k_value, label_map)

    # Output to console:
    for line in output_text:
        print(line)
    print("End of script reached.")
    print("-------------------------------------------------------------------")
