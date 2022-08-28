"""This module handles processing of text and images"""
import numpy as np
import tensorflow as tf

import io_handler
import operation_mode


IMAGE_SHAPE = 224  # By virtue of how the models are trained


def process_image(image: np.ndarray) -> np.ndarray:
    """This function performs input pre-processing for MobileNetV2-based models"""
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)  # cast to 32-bit
    rescaled_image = tf.image.resize(tensor, (IMAGE_SHAPE, IMAGE_SHAPE))  # re-scale to 224 by 224 pixels
    normalised_image = rescaled_image.numpy() / 255  # Normalise values from 0-225 to 0-1
    return normalised_image[np.newaxis]  # Add another dimension: (224, 224, 3) -> (1, 224, 224, 3)
