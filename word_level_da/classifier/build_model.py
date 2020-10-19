"""Module to create model.

Helper functions to create a multi-layer perceptron model and a separable CNN
model. These functions take the model hyper-parameters as input. This will
allow us to create model instances with slightly varying architectures.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import models
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.layers import Concatenate, Input


def rnn_model_fixed(layers,
                    nodes,
                    embedding_dim,
                    dropout_rate,
                    input_shape,
                    num_classes,
                    num_features,
                    use_pretrained_embedding=False,
                    is_embedding_trainable=False,
                    embedding_matrix=None,
                    bidirectional=True,
                    output_bias=None
                    ):
    if output_bias is not None:
        output_bias = initializers.Constant(output_bias)

    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()

    if use_pretrained_embedding:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable,
                            mask_zero=True))
    else:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0]))

    ## Default values to train with GPU 
    for i in range(layers - 1):
        if bidirectional:
            model.add(Bidirectional(LSTM(nodes, activation="tanh", recurrent_activation="sigmoid",
                                         recurrent_dropout=0, use_bias=True,
                                         unroll=False, return_sequences=True)))

        else:
            model.add(LSTM(nodes))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Bidirectional(LSTM(nodes, activation="tanh", recurrent_activation="sigmoid",
                                 recurrent_dropout=0, use_bias=True,
                                 unroll=False, return_sequences=False)))

    model.add(Dropout(dropout_rate))
    model.add(Dense(nodes, activation="relu"))
    model.add(Dropout(dropout_rate))

    model.add(Dense(op_units, activation=op_activation, bias_initializer=output_bias))

    return model


def cnn_model(layers,
              filters,
              filters_size,
              embedding_dim,
              dropout_rate,
              input_shape,
              num_classes,
              num_features,
              use_pretrained_embedding=False,
              is_embedding_trainable=False,
              embedding_matrix=None,
              output_bias=None,
              text_len=64,
              pool_size=1
              ):
    if output_bias is not None:
        output_bias = initializers.Constant(output_bias)

    text_seq_input = Input(shape=(text_len,), dtype='int32')

    text_embedding = Embedding(num_features, embedding_dim, input_length=input_shape[0],
                               weights=[embedding_matrix], trainable=is_embedding_trainable)(text_seq_input)

    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)

    convs = []
    for filter_size in filters_size:
        l_conv = Conv1D(filters=filters, kernel_size=filter_size, padding='valid', activation='relu')(text_embedding)
        # l_pool = MaxPool1D(filter_size)(l_conv)
        l_pool = MaxPool1D(pool_size)(l_conv)
        l_flat = Flatten()(l_pool)
        convs.append(l_flat)

    l_merge = Concatenate(axis=1)(convs)

    l_drop_0 = Dropout(dropout_rate)(l_merge)

    # l_dense = Dense(32, activation='relu')(l_merge)

    # l_drop_1=Dropout(dropout_rate)(l_drop_0)

    l_dense_final = Dense(op_units, activation=op_activation, bias_initializer=output_bias)(l_drop_0)

    model = models.Model(inputs=[text_seq_input], outputs=l_dense_final)

    return model


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation
