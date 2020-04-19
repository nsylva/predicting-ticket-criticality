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


def build_model(seq_len,bert_path, dropout = 0.1):
    #define inputs
    input_word_ids = tf.keras.layers.Input(shape=(seq_len,),dtype=tf.int32,name='input_word_ids')
    input_mask = tf.keras.layers.Input(shape=(seq_len,),dtype=tf.int32,name='input_mask')
    segment_ids = tf.keras.layers.Input(shape=(seq_len,),dtype=tf.int32,name='segment_ids')

    bert_layer = hub.KerasLayer(bert_path,trainable=False,name='bert_layer')
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])


    #define a pooling layer to call between each convolution
    pool = tf.keras.layers.GlobalMaxPool1D()
    
    #convolutions
    cnn1 = tf.keras.layers.Conv1D(filters=64, kernel_size=2, padding='valid', activation='relu')(sequence_output)
    cnn1_pooled = pool(cnn1)
    cnn2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='valid', activation='relu')(sequence_output)
    cnn2_pooled = pool(cnn2)
    cnn3 = tf.keras.layers.Conv1D(filters=64, kernel_size=4, padding='valid', activation='relu')(sequence_output)
    cnn3_pooled = pool(cnn3)

    #concat pooled reps
    concat = tf.concat([cnn1_pooled,cnn2_pooled,cnn3_pooled],axis=-1)
    
    #dense layers
    dense0 = tf.keras.layers.Dense(512,activation='relu',name='dense_0')(concat)
    d0_dropouts = tf.keras.layers.Dropout(dropout)(dense0)
    dense1 = tf.keras.layers.Dense(96,activation = 'relu',name = 'dense_1')(d0_dropouts)
    d1_dropouts = tf.keras.layers.Dropout(dropout)(dense1)
    dense2 = tf.keras.layers.Dense(48,activation = 'relu',name = 'dense_2')(d1_dropouts)
    d2_dropouts = tf.keras.layers.Dropout(dropout)(dense2)
    dense3 = tf.keras.layers.Dense(24,activation = 'relu',name = 'dense_3')(d2_dropouts)
    d3_dropouts = tf.keras.layers.Dropout(dropout)(dense3)

    #output
    out = tf.keras.layers.Dense(4, # We need the output predictions for each of the four categories
                            activation='softmax', # Because we want the probability of each class that ads up to 1
                            name='Prediction')(d3_dropouts)

    return tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x[1]) / float(self.batch_size)))

    def __getitem__(self,idx):
        start_idx = idx*self.batch_size
        end_idx = (idx+1)*self.batch_size

        batch_x = [self.x[0][start_idx:end_idx],self.x[1][start_idx:end_idx],self.x[2][start_idx:end_idx]]
        batch_y = self.y[start_idx:end_idx] #think its this because 1D

        return batch_x, batch_y



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
    print("Initialized.")
    
    #bring in data

    print("Loading training data...")
    train_data = load_json_data_as_array(args.train_data)
    print("Training data loaded.\nLoading test data...")
    test_data = load_json_data_as_array(args.eval_data)
    
    print("Test data loaded. Setting up distributed training strategy...")
    #make tf distribute across GPUs - log off for now for sake of cleanliness
    #tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    
    bert_path = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'
    print("Distributed training strategy ready. Building model...")
    with strategy.scope():
        #build model
        model = build_model(seq_len,bert_path, dropout = args.dropout)
        model.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                    metrics=['accuracy'])

    #initialize logging
   # train_start = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')
   # json_log = open('/project/training_logs/training_log%s.json'%train_start, mode='wt', buffering=1)
   # logging_callback = LogCallback(log_file=json_log)
    print("Model built. Setting up data generator.")
    #data sequence generator
    data_sequence = DataSequence(train_data['x'],train_data['y'], batch_size = args.batch_size)
    print("Data generator ready. Training model...")
    #train model
    model.fit(data_sequence, 
              epochs = args.num_epochs, 
              shuffle = True)
    print("Model trained, saving model...")
    #save trained model
    train_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')
    model.save('/project/models/saved_model_%s.h5'%train_time)          
    print("Model saved, predicting on test data...")
    #predict on model
    predictions = model.predict(test_data['x'])
    print("Done predicting. Save predictions")
    #save predictions to JSON
    pred_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')
    with open('/project/predictions/output_predictions_%s.json'%pred_time,'w') as f:
        json.dump({'predictions': predictions, 'true_labels': test_data['y']},f, cls=NumpyEncoder)
    print("Predictions saved to: /project/predictions/output_predictions_%s.json"%pred_time)
