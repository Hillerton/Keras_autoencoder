"""
script to get activation and shape of layer
"""

import keras
import tensorflow as tf
from keras import layers
import numpy as np

class layer_metrics:
    def layer_outp(model, model_input, layer_name=None):

        from keras import backend as K


        layer_out=[]
        inp = model.input
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]
        funcs = K.function([inp]+ [K.learning_phase()], outputs )

        list_inp = []
        list_inp.extend(model_input)
        activation = []

        layer_outs = funcs([list_inp])
        activation = np.asarray(layer_outs, dtype="float32")
        return (activation)


        """
        #semi working code to pick out activation in each operation within a layer.
        graph = tf.get_default_graph()
        activation=graph.get_operation_by_name('dense_1/MatMul').outputs[0]

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        v = sess.run(activation, {inp:model_input})
        print (v)
        """

    def layer_shape(model, layer_name):

        from keras import layers
        model = model
        shape = [layer.output_shape for layer in model.layers if layer.name == layer_name]
        return (shape)

    def weights(model, layer_name):

        for layer in model.layers:
            if layer.name == layer_name:
                weights = layer.get_weights()
        return weights
