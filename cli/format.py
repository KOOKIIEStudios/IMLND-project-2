"""This module handles processing of text and images"""
import numpy as np
import pandas as pd
import tensorflow as tf

import io_handler
import operation_mode


# Constants that are set by virtue of how the models are trained
IMAGE_SHAPE = 224
RESCALE_FACTOR = 255
NUMBER_OF_CLASSES = 102


def process_image(image: np.ndarray) -> np.ndarray:
    """This function performs input pre-processing for MobileNetV2-based models"""
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)  # cast to 32-bit
    rescaled_image = tf.image.resize(tensor, (IMAGE_SHAPE, IMAGE_SHAPE))  # re-scale to 224 by 224 pixels
    normalised_image = rescaled_image.numpy() / RESCALE_FACTOR  # Normalise values from 0-225 to 0-1
    return normalised_image[np.newaxis]  # Add another dimension: (224, 224, 3) -> (1, 224, 224, 3)


def convert_to_dataframe(prediction: np.ndarray) -> pd.DataFrame:
    # NOTE: The label map starts from index 1!
    dataframe = pd.DataFrame(
        prediction,
        index=range(1, NUMBER_OF_CLASSES + 1),  # associate with labels, in a way that's resistant to sorting
        columns=["probabilities"],
    )
    return dataframe.sort_values(by=["probabilities"], ascending=False)


def filter_top_k_results(prediction: pd.DataFrame, k: int) -> (list[np.float32], list[int]):
    """Filters out the top k results, as a pair of lists

    First list is the probabilities, second list are the numeric labels
    """
    top_k_results = prediction.head(k)
    return top_k_results["probabilities"].values.tolist(), top_k_results.index.tolist()


def basic_output(prediction: pd.DataFrame) -> list[str]:
    buffer: list[str] = []
    probabilities, labels = filter_top_k_results(prediction, 1)
    # There is only one element in each list:
    buffer.append(f"This flower is most likely: {labels[0]}")
    buffer.append(f"    Probability: {probabilities[0]:.2%}")
    return buffer


def top_k_output(prediction: pd.DataFrame, k_value: int) -> list[str]:
    buffer: list[str] = []
    probabilities, labels = filter_top_k_results(prediction, k_value)
    buffer.append(f"Here are the {k_value} most-likely results (from most- to least-likely):")
    for iteration in range(k_value):
        buffer.append(f"    {iteration}. Label: {labels[iteration]}, Likelihood: {probabilities[iteration]:.2%}")
    return buffer


def labeled_top_output(prediction: pd.DataFrame, label_map: dict[str, str]) -> list[str]:
    buffer: list[str] = []
    probabilities, labels = filter_top_k_results(prediction, 1)
    label_name = label_map.get(str(labels[0]))
    # There is only one element in each list:
    buffer.append(f"This flower is most likely: {label_name}")
    buffer.append(f"    Probability: {probabilities[0]:.2%}")
    return buffer


def labeled_top_k_output(
    prediction: pd.DataFrame,
    k_value: int,
    label_map: dict[str, str],
) -> list[str]:
    buffer: list[str] = []
    probabilities, label_numbers = filter_top_k_results(prediction, k_value)
    label_names = [label_map.get(str(element)) for element in label_numbers]
    buffer.append(f"Here are the {k_value} most-likely results (from most- to least-likely):")
    for iteration in range(k_value):
        buffer.append(f"    {iteration}. Label: {label_names[iteration]}, Likelihood: {probabilities[iteration]:.2%}")
    return buffer


def format_output(
    mode: int,
    prediction: pd.DataFrame,
    k_flag: int | None,
    label_map: dict[str, str] | None,
) -> list[str]:
    match mode:
        case operation_mode.Mode.NO_ARGS:
            return basic_output(prediction)
        case operation_mode.Mode.TOP_K:
            return top_k_output(prediction, k_flag)
        case operation_mode.Mode.CATEGORY:
            return labeled_top_output(prediction, label_map)
        case operation_mode.Mode.BOTH:
            return labeled_top_k_output(prediction, k_flag, label_map)
        case _:
            return [
                "We have no idea how you got here, but you did.",
                "Congratulations, you have broken the script.",
            ]
