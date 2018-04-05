"""
script to launch a simple keras autoencoder script.
requiers data input either as biallelic or onehot encoding.
Further requieres scripts; keras_autoencoder_model.py, one_hot_encoder.py, make_log.py and record_TF_metadata.py
Imports:
argparse 1.1
keras 2.1.3
(further imports are standard libraries without version)
made in python 3.3.6
"""

import glob
import argparse
import time
import os
from make_log import make_log
import numpy as np

today = time.strftime("%Y-%m-%d")

parser = argparse.ArgumentParser(description="Script to run an autoencoder on one hot encoded or biallelic genetic SNV data")
parser.add_argument("train_file", action="store",
                    help="Gives the program the name of the name of the bed/bim/fam files for the traning data. Doesn\"t need the the file extension only name")
parser.add_argument("test_file", action="store",
                    help="Gives the program the name of the name of the bed/bim/fam files for the test data. Doesn\"t need the the file extension only name")
parser.add_argument("hidden_nodes", action="store", type=int,
                    help="Dictates how many hidden nodes should be created for the run")
parser.add_argument("epochs", action="store", type=int, default = 1000,
                    help="Dictates how many forward passes the model should perform when traning (default is 1000)")
parser.add_argument("--noise", dest="noise", action="store", type=float, default = 0.0,
                    help="Dictates how much noise should be added to the model by setting input as 0 with n percentage chanse for each input. Should be given as a float of 0.X (defaults to 0.0)")
parser.add_argument("--l1", dest="l1", action="store", type=float, default = 0.00,
                    help="Set the l1_lambda value to make the model sparse. should be given as a float of 0.X (deafult is 0.0 meaning the model is not sparse)")
parser.add_argument("--out", dest="out_path", action="store", default = os.getcwd()+"/"+today+"_autoencoder_run",
                    help="Tells the program where to store output data. Should be given as a directory, and not a file (defaults to a map in current working directory)")
parser.add_argument("--log", dest="log", action="store", default = today+"_AE_log.txt",
                    help="Set file name for the log file storing information and errors from the run. (defaults to date_AE_log.txt)")
parser.add_argument("--log_path", dest="log_path", action="store", default = False,
                    help="Set file path for lof file. (defaults to output direcotry)")
parser.add_argument("--subset", dest="sub", action="store", type=int, default = False,
                    help="Select number of samples to subselect from if needed.")

args = parser.parse_args()

out_path=args.out_path+"/"+today+"_ae_test"

if args.log_path == False:
    log_dir = out_path
else :
    log_dir = args.log_path

if not glob.glob(out_path):
    os.makedirs(out_path)

make_log(args,log_dir)

log_file = open(args.log, "a")

from one_hot_encoder import biallelic_to_onehot
from keras_autoencoder_model import keras_autoencoder
import keras

print("Finding traning data")
traning_data = biallelic_to_onehot.onehot(args.train_file, args.sub)
print ("Finding test data")
test_data = biallelic_to_onehot.onehot(args.test_file, args.sub)

train = round(traning_data.shape[1]*0.8)

idx = np.random.choice(traning_data.shape[1], train, replace=False)

x_train = traning_data[:, idx]
x_val = np.delete(traning_data, idx, 1)

x_train = np.swapaxes(x_train, 0 ,1)
x_val = np.swapaxes(x_val, 0, 1)

tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False, embeddings_metadata=True)

ae_model = keras_autoencoder(3, traning_data.shape[0], args.hidden_nodes, args.l1, args.noise)
"""
y_train = test_data[:, idx]
y_val = np.delete(test_data, idx, 1)

y_train = np.swapaxes(y_train, 0 ,1)
y_val = np.swapaxes(y_val, 0, 1)
"""

traning_data = np.swapaxes(traning_data, 1, 0)
test_data = np.swapaxes(test_data, 1, 0)

ae_model.fit_model(traning_data, test_data, args.epochs, tb)
ae_model.save(out_path)

log_file.close
