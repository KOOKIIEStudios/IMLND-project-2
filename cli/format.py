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
    dataframe = pd.DataFrame(
        prediction,
        index=range(1, NUMBER_OF_CLASSES + 1),  # associate with labels, in a way that's resistant to sorting
        columns=["probabilities"],
    )
    return dataframe.sort_values(by=["probabilities"], ascending=False)
