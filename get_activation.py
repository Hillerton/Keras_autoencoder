import keras.backend as K
import keras
import numpy as np
from keras.models import load_model
from get_layer_metrics import layer_metrics
from one_hot_encoder import biallelic_to_onehot

def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    #print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            pass
            #print(layer_activations.shape)
        else:
            pass

    return activations

path = "/home/hillerton/results/2018-03-29_ae_keras_run/deep_ae/regression/"
model = path+"regression_model.h5"
data = "/home/hillerton/Data/intersect_1000g_cancer/212_cancer/bed_files/"

ae_model = load_model(model)

bed_data, bim = biallelic_to_onehot.onehot(data)
bed_data = np.swapaxes(bed_data, 0, 1)

activation = get_activations(ae_model, bed_data, layer_name="leaky_re_lu_1")

fil = open(path+"leaky_relu_activation.csv", "a+")

for i in activation:
    np.savetxt(fil, i, delimiter="\t")
