#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import nose
import os
import time
from text_processor.text_processor import KerasTextListPreparer
import keras.callbacks
import keras_model.model_builder as ModelBuilder
from keras_model.keras_model import LSTMModel,WordEmbedder
from keras_model.model_builder import ModelBuilder as ModelBuilder


class TestModelBuilder(unittest.TestCase):
    
    def setUp(self):
        self.conf = {'corpus_filename':'unit_test_corpus.pkl',
                   'text_proc_params':{
                           'sentence_start_token':'SENTENCE_START',
                           'sentence_end_token':'SENTENCE_END',
                           'unknown_token':'UNKNOWN_TOKEN'
                           },
                   'word_embedding_params':{
                           'embedding_directory':'/Volumes/GLOVE/glove.6B',
                           'embedding_file':'glove.6B.50d.txt'
                           },      
                   'model_params':{
                           'vocab_size':10,
                           'num_timesteps': 6,
                           'validation_split':.5,
                           'embed_dim':100,
                           'use_pretrained_word_embeddings':False
                           }}

        self.model_builder_obj = ModelBuilder(
                                            self.conf,
                                            KerasTextListPreparer,
                                            LSTMModel,
                                            WordEmbedder)
        
    def test_model_builder_object_returns_constructor_params(self):        
        self.assertEqual(self.model_builder_obj.corpus_filename, 'unit_test_corpus.pkl')
        self.assertEqual(
                        self.model_builder_obj.text_proc_params,
                         {'sentence_start_token':'SENTENCE_START',
                           'sentence_end_token':'SENTENCE_END',
                           'unknown_token':'UNKNOWN_TOKEN'
                           })              
        self.assertEqual(
                        self.model_builder_obj.model_params,
                         {'vocab_size':10,
                           'num_timesteps': 6,
                           'validation_split':.5,
                           'embed_dim':100,
                           'use_pretrained_word_embeddings':False
                           })
        self.assertEqual(self.model_builder_obj.word_embedding_params,
                         {'embedding_directory':'/Volumes/GLOVE/glove.6B',
                          'embedding_file':'glove.6B.50d.txt'
                          })                       
        self.assertNotEqual(self.model_builder_obj.lstm_model.__dict__,             
                         LSTMModel(
                                   self.model_builder_obj.x_train_all_samples,
                                   self.model_builder_obj.y_train_all_samples,
                                   vocab_size = 10,
                                   num_timesteps = 6,
                                   validation_split = .5,
                                   embed_dim = 100).__dict__)
    def test_load_corpus(self):
        text_list_test = self.model_builder_obj.load_corpus()
        self.assertEqual(text_list_test,
                         [
                          'Hello there, good sir! \n.', 
                          'The park is \n\t is one block to the west, sir.'
                          ])

    def test_train_model(self):
        self.assertIsInstance(self.model_builder_obj.trained_lstm_model,keras.callbacks.History)
    
    def test_adding_embed_matrix_updates_model_params(self):
        embed_conf = {'corpus_filename':'unit_test_corpus.pkl',
                   'text_proc_params':{
                           'sentence_start_token':'SENTENCE_START',
                           'sentence_end_token':'SENTENCE_END',
                           'unknown_token':'UNKNOWN_TOKEN'
                           },
                   'word_embedding_params':{
                           'embedding_directory':'/Volumes/GLOVE/glove.6B',
                           'embedding_file':'glove.6B.50d.txt'
                           },      
                   'model_params':{
                           'vocab_size':10,
                           'num_timesteps': 6,
                           'validation_split':.5,
                           'embed_dim':50,
                           'use_pretrained_word_embeddings':True
                           }}

        model_builder_with_pretrained_emb_obj = ModelBuilder(
                                                    embed_conf,
                                                    KerasTextListPreparer,
                                                    LSTMModel,
                                                    WordEmbedder)
        self.assertTrue(
                'embedding_matrix' in model_builder_with_pretrained_emb_obj.model_params.keys())
        
    def test_create_filename(self):
        filename_test = self.model_builder_obj.create_filename('model_results.pickle')
        self.assertEqual(filename_test,time.strftime("%d_%m_%Y") + '_' + 'model_results.pickle')

    @staticmethod
    def remove_file_from_dir(filename_input):
        if os.path.exists(filename_input):
            os.remove(filename_input)
            
    def test_save_training_data(self):
        filename_x = time.strftime("%d_%m_%Y") + '_' + 'x_train_all_samples.npy'
        filename_y = time.strftime("%d_%m_%Y") + '_' + 'y_train_all_samples.npy'
        self.remove_file_from_dir(filename_x)
        self.remove_file_from_dir(filename_y)
        self.model_builder_obj.save_training_data()
        self.assertTrue(os.path.exists(filename_x))
        self.assertTrue(os.path.exists(filename_y))

    def test_save_word_to_index_dict(self):
        filename_dict = time.strftime("%d_%m_%Y") + '_' + 'word_to_index_dict.pkl'
        self.remove_file_from_dir(filename_dict)
        self.model_builder_obj.save_word_to_index_dict()
        self.assertTrue(os.path.exists(filename_dict))
        
    def test_save_model(self):
        filename_path = time.strftime("%d_%m_%Y") + '_'+ 'text_generation_model.h5'
        self.remove_file_from_dir(filename_path)
        self.model_builder_obj.save_model()
        self.assertTrue(os.path.exists(filename_path))

    def test_save_model_results(self):
        current_date = time.strftime("%d_%m_%Y")
        filename_path_results = current_date+'_model_results.pickle'
        self.remove_file_from_dir(filename_path_results)
        self.model_builder_obj.save_model_results()
        self.assertTrue(os.path.exists(filename_path_results))
        
        
if __name__=='__main__':
    nose.run(defaultTest=__name__)
