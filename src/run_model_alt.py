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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_json_data_as_array(path):
    with open(path) as f:
        data = json.load(f)
        ids, masks, segments = np.asarray(data['data'][0]),np.asarray(data['data'][1]), np.asarray(data['data'][2])
        x = [ids, masks, segments]
    return {'x': x, 'y': np.asarray(data['labels'])}


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
'''
#define a callback for reporting training results
class LogCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
    
    def on_epoch_end(self, epoch, logs = {}):
        data = {'epoch': epoch, 'loss': logs['loss'], 'accuracy': logs['acc']}
        self.log_file.write(json.dumps(data + '\n'))
    
    def on_train_end(self):
        self.log_file.close()
'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--train_data', 
                        type = str, 
                        help = 'Path to training data.')
    parser.add_argument('-e',
                        '--eval_data', 
                        type = str, 
                        help = 'Path to test data.')
    parser.add_argument('-l',
                        '--seq_len',
                        type = int,
                        help = 'Sequence length of input data. Defaults to 128, max 512.',
                        default = 128)
    parser.add_argument('-n',
                        '--num_epochs',
                        type = int,
                        help = 'Number of epochs to train for.',
                        default = 1)
    parser.add_argument('-b',
                        '--batch_size',
                        type = int,
                        help = 'Number of examples per batch',
                        default = 32)
    parser.add_argument('-d',
                        '--dropout',
                        type = float,
                        help = 'Fraction dropout. Default = 0.1',
                        default = 0.1)

    args = parser.parse_args()

    if args.seq_len > 512 or args.seq_len < 1:
        raise ValueError('Sequence Length cannot be greater than 512 must be at least 1.')
    else:
        seq_len = args.seq_len
    
    #bring in data
    train_data = load_json_data_as_array(args.train_data)
    test_data = load_json_data_as_array(args.eval_data)

    #make tf distribute across GPUs - log off for now for sake of cleanliness
    #tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()

    bert_path = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'
    
    with strategy.scope():
        #build model
        model = build_model(seq_len, bert_path, dropout = args.dropout)
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['sparse_categorical_accuracy'])

    #initialize logging
   # train_start = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')
   # json_log = open('/project/training_logs/training_log%s.json'%train_start, mode='wt', buffering=1)
   # logging_callback = LogCallback(log_file=json_log)

    #train model
    model.fit(train_data['x'],
              train_data['y'], 
              epochs = args.num_epochs, 
              batch_size = args.batch_size, 
              shuffle = True)

    #save trained model
    train_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')
    model.save('/project/models/saved_model_%s.h5'%train_time)          

    #predict on model
    predictions = model.predict(test_data['x'])

    #save predictions to JSON
    pred_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')
    with open('/project/predictions/output_predictions_%s.json'%pred_time,'w') as f:
        json.dump({'predictions': predictions, 'true_labels': test_data['y']},f, cls=NumpyEncoder)

