
import keras
import numpy as np
from keras.models import load_model
from get_layer_metrics import layer_metrics
from one_hot_encoder import biallelic_to_onehot
from get_activation import get_activations
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path = "/home/hillerton/results/2018-04-05_ae_test/"
model = path+"AE_model.h5"

ae_model = load_model(model)


weight = layer_metrics.weights(ae_model, "dense_1")
#np.savetxt(path+"weights.csv", weight[0], delimiter="\t")


for line in weight[0]:
    count = 0
    aver_mean = np.mean(line)
    std_err = np.std(line)
    up_cut = aver_mean + 2 * std_err
    low_cut = aver_mean - 2 * std_err
    for i in line:
        if i > up_cut or i < low_cut:
            pass

#weight = np.swapaxes(weight[0], 1, 0)

"""
pca = PCA(n_components=3, svd_solver="randomized")
out = pca.fit_transform(weight[0])
print (out.shape)

colors = ['navy', 'turquoise', 'darkorange']
plt.figure(figsize=(8, 8))
plt.scatter(out[:, 0], out[:, 1], out[:, 2], color=colors)
plt.show()
"""
