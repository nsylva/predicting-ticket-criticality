#libraries
import bert
import datetime
import h5py
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import  Model

import pydot
import graphviz

def build_model(seq_len, bert_path, dropout = 0.1):
    #define inputs
    input_word_ids = tf.keras.layers.Input(shape=(seq_len,), 
                                           dtype=tf.int32,
                                           name='input_word_ids')
    input_mask = tf.keras.layers.Input(shape=(seq_len,), 
                                       dtype=tf.int32,
                                       name='input_mask')

    segment_ids = tf.keras.layers.Input(shape=(seq_len,), 
                                        dtype=tf.int32,
                                        name='segment_ids')
    
    # bert
    bert_layer = hub.KerasLayer(bert_path, 
                                    trainable=True,
                                    name='bert_layer')
    #feed inputs to bert layer
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    #dropout
    dropouts = tf.keras.layers.Dropout(dropout)(pooled_output)

    #output
    out = tf.keras.layers.Dense(4, # We need the output predictions for each of the four categories
                            activation='softmax', # Because we want the probability of each class that ads up to 1
                            name='Prediction')(dropouts)

    return tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--seq_len',
                        type = int,
                        help = 'Sequence length of input data. Defaults to 128, max 512.',
                        default = 128)

    args = parser.parse_args()

    if args.seq_len > 512 or args.seq_len < 1:
        raise ValueError('Sequence Length cannot be greater than 512 must be at least 1.')
    else:
        seq_len = args.seq_len
    print("Initialized.")
    
    #bring in data

    
    print("Test data loaded. Setting up distributed training strategy...")
    #make tf distribute across GPUs - log off for now for sake of cleanliness
    #tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()

    bert_path = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'
    print("Distributed training strategy ready. Building model...")
    with strategy.scope():
        #build model
        model = build_model(seq_len, bert_path, dropout = 0.1)
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    
    model.summary()
    tf.keras.utils.plot_model(model, '/project/model_diagrams/model_1.png', show_shapes=True)
