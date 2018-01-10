#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import nose
import numpy as np
from keras.models import Sequential
import keras.callbacks
import re
import keras_model.keras_model as km


class TestLSTMModel(unittest.TestCase):    
    
    def setUp(self):
        self.x  =  np.array(
                [
                [1, 24, 14, 83, 2500, 7, 81, 118],
                [1, 2500, 8, 2041, 0, 0, 0, 0],
                [1, 127, 35, 55, 70, 90, 356, 608],
                [1, 1734, 4, 2500, 0, 0, 0, 0],
                [1, 13, 106, 5, 14, 384, 22, 89]
                ])
        self.y  =  np.array([
                [24, 14, 83, 2500, 7, 81, 118, 4],
                [2500, 8, 2041, 2, 0, 0, 0, 0],
                [127, 35, 55, 70, 90, 356, 608, 47],
                [1734, 4, 2500, 2, 0, 0, 0, 0],
                [13, 106, 5, 14, 384, 22, 89, 37]
                ])
        self.lstm_model_obj  =  km.LSTMModel(
                                         x = self.x,
                                         y = self.y,
                                         vocab_size = 8000,
                                         num_timesteps = 8,
                                         num_epochs = 1,
                                         validation_split = .4,
                                         embed_dim = 200,
                                         mask_zero = True,
                                         lstm_hidden_dim = 250,
                                         use_pretrained_word_embeddings = False)
        self.lstm_model_obj.compile_model()                      #for testing save methods
        self.lstm_model_architecture = self.lstm_model_obj.build_model_architecture()

    def test_lstm_model_object_returns_constructor_params(self):
        self.assertTrue(np.array_equal(self.lstm_model_obj.x,self.x))
        self.assertTrue(np.array_equal(self.lstm_model_obj.y,self.y))
        self.assertEqual(self.lstm_model_obj.vocab_size,8000)
        self.assertEqual(self.lstm_model_obj.num_timesteps,8)
        self.assertEqual(self.lstm_model_obj.batch_size,128)
        self.assertEqual(self.lstm_model_obj.num_epochs,1)
        self.assertEqual(self.lstm_model_obj.validation_split,.4)
        self.assertEqual(self.lstm_model_obj.embed_dim,200)
        self.assertEqual(self.lstm_model_obj.lstm_hidden_dim,250)
        self.assertEqual(self.lstm_model_obj.use_pretrained_word_embeddings,False)

    def test_build_model_architecture(self):
        self.assertIsInstance(self.lstm_model_architecture,Sequential)

    def test_output_model_layer_description(self):
        layer_descriptions_list = [[
                                  str(e) for e 
                                  in self.lstm_model_architecture.layers[x].trainable_weights] 
                                  for x in range(len(self.lstm_model_architecture.layers))]   
        def extract_layer_shape(layer_description_string):
            layer_shape = re.findall(r'\d+', layer_description_string.split('=')[1])
            layer_shape = list(map(int,layer_shape))
            return layer_shape
        kernel_layer_shape = extract_layer_shape(layer_descriptions_list[1][0])              
        recurrent_kernel_layer_shape = extract_layer_shape(layer_descriptions_list[1][1])              
        output_layer_shape = extract_layer_shape(layer_descriptions_list[2][0])
        self.assertEqual(
                kernel_layer_shape,
                [self.lstm_model_obj.embed_dim,self.lstm_model_obj.lstm_hidden_dim*4])
        self.assertEqual(
                recurrent_kernel_layer_shape,
                [self.lstm_model_obj.lstm_hidden_dim,self.lstm_model_obj.lstm_hidden_dim*4])
        self.assertEqual(
                output_layer_shape,
                [self.lstm_model_obj.lstm_hidden_dim,self.lstm_model_obj.vocab_size+1])

    def test_get_model(self):
        lstm_test_model = self.lstm_model_obj.get_model()
        self.assertIsInstance(lstm_test_model,Sequential)

    def test_copy_training_data(self):
        x_copy,y_copy = self.lstm_model_obj.copy_training_data()
        self.assertNotEqual(id(self.x),id(x_copy))
        self.assertTrue(np.array_equal(self.x,x_copy))
        self.assertNotEqual(id(self.y),id(y_copy))
        self.assertTrue(np.array_equal(self.y,y_copy))

    def test_shuffle_data(self):
        data_shuffled = self.lstm_model_obj.shuffle_data(self.x)
        aset = set([tuple(z) for z in data_shuffled])      
        bset = set([tuple(z) for z in self.x])
        num_common_rows = np.array([z for z in aset & bset]).shape[0]
        self.assertEqual(num_common_rows,self.x.shape[0])
        self.assertFalse(np.array_equal(data_shuffled,self.x))

    def test_fit_model(self):
        self.assertIsInstance(self.lstm_model_obj.fit_model(),keras.callbacks.History)

    def test_create_data_validation_split(self):
        x_train,y_train,x_val,y_val = self.lstm_model_obj.create_data_validation_split()
        self.assertTrue(np.array_equal(
                        x_train,
                        [
                        [1, 24, 14, 83, 2500, 7, 81, 118],
                        [1, 127, 35, 55, 70, 90, 356, 608],
                        [1, 2500, 8, 2041, 0, 0, 0, 0]
                        ]
                         ))
        self.assertTrue(np.array_equal(
                      y_train,
                      [[[24],[14],[83], [2500],[7],[81],[118],[4]],
                      [[127],[35],[55],[70],[90],[356],[608],[47]],
                      [[2500],[8],[2041],[2],[0],[0],[0],[0]]
                      ]
                       ))
        self.assertTrue(np.array_equal(
                x_val,
                [[1, 1734, 4, 2500, 0, 0, 0, 0],
                [1, 13, 106, 5, 14, 384, 22, 89]]
                ))

        self.assertTrue(np.array_equal(
                y_val,
                [[[1734],[4],[2500],[2],[0],[0],[0],[0]],
                [[13],[106],[5],[14],[384],[22],[89],[37]]
                ]
                ))
        
    def test_check_pretrained_word_embeddings_inputs_match(self):
        self.assertRaises(
                ValueError,
                km.LSTMModel,
                **{'x':self.x,
                   'y':self.y,
                   'vocab_size':8000,
                   'num_timesteps':8,
                   'num_epochs':1,
                   'validation_split':.4,
                   'embed_dim':200,
                   'mask_zero':True,
                   'lstm_hidden_dim':250,
                   'use_pretrained_word_embeddings':False,
                   'embedding_matrix':[[1,1,2,3,2],[4,6,2,4,5]]
                   })    
        self.assertRaises(
                ValueError,
                km.LSTMModel,
                **{'x':self.x,
                   'y':self.y,
                   'vocab_size':8000,
                   'num_timesteps':8,
                   'num_epochs':1,
                   'validation_split':.4,
                   'embed_dim':200,
                   'mask_zero':True,
                   'lstm_hidden_dim':250,
                   'use_pretrained_word_embeddings':True,
                   'embedding_matrix':None
                   })    

  
class TestWordEmbedder(unittest.TestCase):
        
    def setUp(self):
        self.embedding_directory = '/Volumes/GLOVE/glove.6B'
        self.embedding_file = 'glove.6B.50d.txt'
        self.vocab_size = 10
        self.embed_dim = 100
        self.sentence_start_token = 'SENTENCE_START'
        self.sentence_end_token = 'SENTENCE_END'
        self.unknown_token = 'UNKNOWN_TOKEN'
        self.word_to_index_dict = {'an': 1, 'SENTENCE_START': 2, 'stopped': 7, 
                              'the': 3, 'assessment': 9, 'SENTENCE_END': 4, 
                              'UNKNOWN_TOKEN': 10, 'governor': 8, 'could': 5,
                              'have': 6
                              }
        self.word_embedder_obj = km.WordEmbedder(
                                  self.embedding_directory,
                                  self.embedding_file,
                                  self.vocab_size,
                                  self.embed_dim,
                                  self.sentence_start_token,
                                  self.sentence_end_token,
                                  self.unknown_token,
                                  self.word_to_index_dict)
    
    def test_word_embedder_returns_constructor_params(self):
        self.assertTrue(
                self.embedding_directory,
                self.word_embedder_obj.embedding_directory)
        self.assertTrue(
                self.embedding_file,
                self.word_embedder_obj.embedding_file)
        self.assertEqual(self.word_embedder_obj.vocab_size,10)
        self.assertTrue(self.word_embedder_obj.embed_dim,100)
        self.assertTrue(self.word_embedder_obj.sentence_start_token,'SENTENCE_START')
        self.assertTrue(self.word_embedder_obj.sentence_end_token,'SENTENCE_END')   
        self.assertTrue(self.word_embedder_obj.unknown_token,'UNKNOWN_TOKEN')           
        self.assertEqual(self.word_embedder_obj.word_to_index_dict,self.word_to_index_dict)

    def test_create_word_to_embeddings_dict(self):
        word_to_embeddings_dict = self.word_embedder_obj.create_word_to_embeddings_dict()   
        self.assertIsInstance(word_to_embeddings_dict,dict)
        self.assertTrue(
                np.allclose(
                        word_to_embeddings_dict['dedicate'][:5],
                        np.asarray([0.29185, 0.30204001, 0.36657, -0.79457998,  0.94717997])))
        self.assertTrue(
                np.allclose(
                        word_to_embeddings_dict['daily'][:5],
                        np.asarray([-0.0013649, 0.68932998, -0.15317, -0.27924001, -0.55024999])))
            
    def test_check_word_embedding_dimension_correct(self):
        self.assertRaises(
                ValueError,
                self.word_embedder_obj.check_word_embedding_dimension_correct,
                'the')    
        
    def test_create_init_vectors_for_tokens(self):
        init_vec = self.word_embedder_obj.create_init_vectors_for_tokens(121) 
        self.assertTrue(np.allclose(init_vec[0][:4],[
                0.11133083, 0.21076757, 0.23296249, 0.15194456
                ]))
        self.assertEqual(init_vec.shape[1],self.word_embedder_obj.embed_dim)      

    def test_create_embedding_matrix(self):
        new_embedding_file_correct_dim = 'glove.6B.100d.txt'
        word_embedder_obj = km.WordEmbedder(
                             self.embedding_directory,
                             new_embedding_file_correct_dim,
                             self.vocab_size,
                             self.embed_dim,
                             self.sentence_start_token,
                             self.sentence_end_token,
                             self.unknown_token,
                             self.word_to_index_dict)
        embedding_matrix = word_embedder_obj.create_embedding_matrix()
        start_token_vec = word_embedder_obj.create_init_vectors_for_tokens(121)
        end_token_vec = word_embedder_obj.create_init_vectors_for_tokens(243)
        unknown_token_vec = word_embedder_obj.create_init_vectors_for_tokens(377)        

        self.assertTrue(np.array_equal(embedding_matrix[2,],start_token_vec[0]))
        self.assertTrue(np.array_equal(embedding_matrix[4,],end_token_vec[0]))
        self.assertTrue(np.array_equal(embedding_matrix[10,],unknown_token_vec[0]))

        
if __name__=='__main__':
    nose.run(defaultTest = __name__)
