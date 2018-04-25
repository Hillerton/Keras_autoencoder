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
parser.add_argument("file_name", action="store",
                    help="Gives the program the name of the name of the bed/bim/fam files to look for. Doesn\"t need the the file extension only name")
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
parser.add_argument("--data_type", dest="data_type", action="store", default = "biallelic",
                    help="Set converting from biallelic or load data directly as onehot encoded. Takes two arguments \"biallelic\" or \"onehot\" (default is biallelic conversion)")
parser.add_argument("--log", dest="log", action="store", default = today+"_AE_log.txt",
                    help="Set file name for the log file storing information and errors from the run. (defaults to date_AE_log.txt)")
parser.add_argument("--log_path", dest="log_path", action="store", default = False,
                    help="Set file path for lof file. (defaults to output direcotry)")
parser.add_argument("--model", dest="model", action="store", default = False,
                    help="Set pretrained model to load. Requierd for regression")
parser.add_argument("--get-active", dest="active", action="store", default = False,
                    help="Set pretrained model to load (default is False, new model will be trained))")
parser.add_argument("--subset", dest="sub", action="store", type=int, default = False,
                    help="Select number of samples to subselect from if needed.")
parser.add_argument("--regression", dest="regression", action="store", default=False,
		            help="Determines is a pretraning or a regression model should be done. (values is True or False)")
parser.add_argument("--phenotypes", dest="phenotype", action="store", default=False,
		            help="Determines phenotypes for regression")

args = parser.parse_args()

out_path=args.out_path+"/"+today+"_ae_test"

if args.log_path == False:
    log_dir = out_path
else :
    log_dir = args.log_path

if not glob.glob(out_path):
    os.makedirs(out_path)

#make_log(args,log_dir)

log_file = open(args.log, "a")

in_file = args.file_name

from one_hot_encoder import biallelic_to_onehot
#from deep_keras_autoencoder import keras_autoencoder
from keras_autoencoder_model import keras_autoencoder
import keras
import tensorflow as tf

if args.data_type == "biallelic":
    data, bim = biallelic_to_onehot.onehot(in_file, args.sub)
#elif args.data_type == "onehot":
#    data = numpy.genfromtxt(in_file)
#else:
#    print ("data_type must be sett to \"biallelic\" or \"onehot\", run terminated due to unrecognised data type.\n See help for options", file=log_file)

train = round(data.shape[1]*0.8)

idx = np.random.choice(data.shape[1], train, replace=False)

x_train = data[:, idx]
x_val = np.delete(data, idx, 1)

x_train = np.swapaxes(x_train, 0 ,1)
x_val = np.swapaxes(x_val, 0, 1)

tb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False, embeddings_metadata=True)
tb2 = keras.callbacks.TensorBoard(log_dir=log_dir+"/regression", histogram_freq=1, write_graph=True, write_images=False, embeddings_metadata=True)

if args.regression == False:
    ae_model = keras_autoencoder(3, data.shape[0], args.hidden_nodes, args.l1, args.noise)

    ae_model.fit_model(x_train, x_val, args.epochs, tb)
    ae_model.save(out_path)


if args.regression == "True":
    if not glob.glob(out_path+"/regression"):
        os.makedirs(out_path+"/regression")

    print ("Preforming regression analysis")

    from keras.models import load_model
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn import preprocessing
    import csv

    model = load_model(args.model)
    lpk=[]
    npk=[]
    tpk=[]

    with open(args.phenotype, "r", encoding='utf16') as pheno_fil:
        all_phenotypes=csv.reader(pheno_fil, delimiter="\t")
        next(all_phenotypes)
        for row in all_phenotypes:
            lpk.append(row[11])
            npk.append(row[19])
            tpk.append(row[27])

    e = []
    for i in lpk:
        if i == 'NA':
            e.append(0.0)
        else:
            e.append(i)
    lpk=np.asarray(e, dtype="float32")
    lpk=np.reshape(lpk, (-1, 1))
    #lpk = preprocessing.normalize(lpk, axis=0, norm='max')

    e = []
    for i in npk:
        if i == 'NA':
            e.append(0.0)
        else:
            e.append(i)
    npk=np.asarray(e, dtype="float32")
    npk=np.reshape(npk, (-1, 1))
    #npk = preprocessing.normalize(npk, axis=0, norm='max')

    e = []
    for i in tpk:
        if i == 'NA':
            e.append(0.0)
        else:
            e.append(i)
    tpk=np.asarray(e, dtype="float32")
    tpk=np.reshape(tpk, (-1, 1))
    #tpk = preprocessing.normalize(tpk, axis=0, norm='max')

    labels = np.dstack((lpk, npk, tpk))
    labels = np.swapaxes(labels, 0, 1)
    pheno_out = open(out_path+"/regression/blood_values.csv", 'w+')
    print_label = labels.reshape(-1, labels.shape[-1])
    np.savetxt(pheno_out, print_label, delimiter="\t")
    pheno_out.close

    y_cut = round(labels.shape[1]*0.8)
    idx = np.random.choice(labels.shape[1], y_cut, replace=False)


    y_train = labels[:, idx]
    y_train = y_train.reshape(-1, y_train.shape[-1])
    y_test = np.delete(labels, idx, 1)
    y_test = y_test.reshape(-1, y_test.shape[-1])

    reg_mod = keras_autoencoder.regression_model(model, 3, data.shape[0], args.hidden_nodes, x_train, x_val, y_train, y_test, args.epochs, tb2, out_path)

    x_reg = np.swapaxes(data, 0, 1)
    x_reg = [x_reg]
    score = reg_mod.predict(x_reg, batch_size=1)
    regfile = out_path+"/regression/"+"regression.csv"
    np.savetxt(regfile, score, delimiter="\t")

else:
    exit()

log_file.close
