
import keras
import numpy as np
from keras.models import load_model
from get_layer_metrics import layer_metrics
from one_hot_encoder import biallelic_to_onehot
from get_activation import get_activations

path = "/home/hillerton/results/2018-03-19_ae_test/"
model = path+"regression_model.h5"

ae_model = load_model(model)

in_file = "/home/hillerton/Data/cancer_filtered_bed/"

data = data = biallelic_to_onehot.onehot(in_file)
data = np.swapaxes(data, 1, 0)

activation = layer_metrics.layer_outp(ae_model, data, layer_name="leaky_re_lu_2")
for i in activation:
    for z in i:
        np.savetxt(path+"activation.csv", z, delimiter="\t")

weight = layer_metrics.weights(ae_model, "dense_4")
for x in weight:
    for z in x:
        np.savetxt(path+"weights.csv", z, delimiter="\t")

#layer_shape = layer_metrics.layer_shape(ae_model, "dense_2")
#print (layer_shape)
