"""
Script to create function to get metadata from tensorflow with keras api.
executed from init_autoencoder.py
"""

import tensorflow
from keras.callbacks import TensorBoard
from tensorflow.python.client import timeline

class TB(TensorBoard):
    def __init__()
