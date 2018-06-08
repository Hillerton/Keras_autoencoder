"""
script to get weight matrix from keras model
"""

import keras
from keras.models import load_model
from keras.utils import plot_model

import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Script to run an autoencoder on one hot encoded or biallelic genetic SNV data")
parser.add_argument("model", action="store",
                    help="Gives the program the model")
parser.add_argument("out", action="store",
                    help="Gives the program the output dir")

args = parser.parse_args()

ae_model = load_model(args.model)

weight = ae_model.get_weights()

with open(args.out+"/weights.csv", "w+") as fil:
    for row in weight:
        np.savetxt(fil, row, delimiter="\t")
