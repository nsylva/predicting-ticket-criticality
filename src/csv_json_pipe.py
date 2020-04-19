import argparse
import bert
import csv
import json
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub

from multiprocessing import Pool
from functools import partial

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Pipeline(object):
    def __init__(self, csv_filepath, sequence_length):
        self.csv_filepath = csv_filepath
        self.csv_file = open(self.csv_filepath,'r')
        self.csv_reader = csv.reader(self.csv_file,delimiter=',')
        self.csv_data = {'train': [], 'test' :[]}
        
        print('Importing CSV data...')
        for line_no,line in enumerate(self.csv_reader): #ignore headers
            if line_no == 0:
                continue

            if line[-1] == 'train':
                self.csv_data['train'].append(line)
            elif line[-1] =='test':
                self.csv_data['test'].append(line)
            else:
                print('Last element of line is not valid. Line #: %s. Value: %s'%(line_no,line[-1]))
        
        print('CSV data imported.')

        self.sequence_length = sequence_length
        self.bert_path = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'
        self.tokenizer = self.build_tokenizer()
        #use all but 1 thread to support other concurrently running tasks
        self.num_processes = os.cpu_count() - 1

        print('Pipeline initialized.')

    def build_tokenizer(self):
        print('Building tokenizer...')
        bert_layer = hub.KerasLayer(self.bert_path, trainable = False, name = 'BERT')
        FullTokenizer = bert.bert_tokenization.FullTokenizer
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        print('Built tokenizer.')
        return FullTokenizer(vocab_file, do_lower_case)

    def create_x_data(self):
        print('Creating input data using %i processors...'%self.num_processes)
        train_sentences = [record[4] for record in self.csv_data['train']]
        test_sentences = [record[4] for record in self.csv_data['test']]
        with Pool(self.num_processes) as p:
            train_ids, train_masks, train_segments = zip(*p.map(partial(create_single_input,tokenizer=self.tokenizer,sequence_length=self.sequence_length),train_sentences))
            test_ids, test_masks, test_segments = zip(*p.map(partial(create_single_input,tokenizer=self.tokenizer,sequence_length=self.sequence_length),test_sentences))

        self.train_x = [np.vstack(train_ids), np.vstack(train_masks), np.vstack(train_segments)]
        self.test_x = [np.vstack(test_ids), np.vstack(test_masks), np.vstack(test_segments)]
        print('Input data created.')

    def create_y_data(self):
        print('Creating output data...')
        self.train_y = np.vstack([np.array([int(record[5]),int(record[6]),int(record[7]),int(record[8])]) for record in self.csv_data['train']])
        self.test_y = np.vstack([np.array([int(record[5]),int(record[6]),int(record[7]),int(record[8])]) for record in self.csv_data['test']])
        print('Output data created.')

    def dump_json_data(self):
        print('Dumping data to JSON.')
        train_data_to_dump = {
            'data' : self.train_x,
            'labels' : self.train_y
        }

        test_data_to_dump = {
            'data' : self.test_x,
            'labels' : self.test_y
        }

        with open(self.csv_filepath[:-4] + '_train.json', 'w') as f:
            json.dump(train_data_to_dump, f, cls=NumpyEncoder)
        print('Dumped training data to file: '+ self.csv_filepath[:-4] + '_train.json')
        with open(self.csv_filepath[:-4] + '_test.json','w') as f:
            json.dump(test_data_to_dump, f, cls=NumpyEncoder)
        print('Dumped test data to file: '+ self.csv_filepath[:-4] + '_test.json')
        

def get_masks(tokens, sequence_length):
    return np.array([1]*len(tokens) + [0] * (sequence_length - len(tokens)))

def get_segments(tokens,sequence_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.array(segments + [0] * (sequence_length - len(tokens)))

def get_ids(tokens, sequence_length, tokenizer):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (sequence_length-len(token_ids))
    return np.array(input_ids)

def create_single_input(sentence, tokenizer, sequence_length):
    stokens = tokenizer.tokenize(sentence)
    stokens = stokens[:sequence_length-2]
    stokens = ["[CLS]"] + stokens + ["[SEP]"]
    
    ids = get_ids(stokens,sequence_length,tokenizer)
    masks = get_masks(stokens,sequence_length)
    segments = get_segments(stokens,sequence_length)

    return ids, masks, segments

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data', 
                        type = str, 
                        help = 'Path to csv data.')
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

    pipeline = Pipeline(args.data,seq_len)
    pipeline.create_x_data()
    pipeline.create_y_data()
    pipeline.dump_json_data()
