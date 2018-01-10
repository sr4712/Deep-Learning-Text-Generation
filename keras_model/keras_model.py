#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABCMeta,abstractmethod
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, TimeDistributed


class LanguageModel(metaclass = ABCMeta):
    
    ''' Abstract Base Class to help build keras Language Model for text generation
    
    
    # Arguments:
        x: numpy array containig training data
        y: numpy array containing labels
        vocab_size: vocab_size for keras model
        num_timesteps: number of timesteps in model
        batch_size: number of samples for every gradient update (int)
        num_epochs: number of epochs to use for training (int)
        validation_split: fraction of data that should be reserved
                          for validating model when building it 
                          (float between 0 and 1)
        optimizer: optimizer to use when minizing loss function
        loss: type of loss to use when building model
        metrics: metric to use when building model        
        '''
            
    def __init__(
                self,
                 x,
                 y,
                 vocab_size,
                 num_timesteps,
                 batch_size = 128,
                 num_epochs = 1,
                 validation_split = .2,
                 optimizer = 'rmsprop',
                 loss = 'sparse_categorical_crossentropy',
                 metrics = 'sparse_categorical_accuracy'):
        
        self.x = x
        self.y = y
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.training_model = self.get_model()

    @abstractmethod
    def build_model_architecture(self):
        return

    def get_model(self):
        self.training_model = self.build_model_architecture()
        return self.training_model

    def compile_model(self):
        self.training_model.compile(loss = self.loss,
                                    optimizer = self.optimizer, 
                                    metrics = [self.metrics])
        
    def copy_training_data(self):
        return(self.x.copy(),self.y.copy())

    def shuffle_data(self,data_input):
        np.random.seed(421)
        num_values = data_input.shape[0]
        val_indices = np.arange(num_values)
        np.random.shuffle(val_indices)
        return(data_input[val_indices])

    def create_data_validation_split(self):
        x_train_all_samples,y_train_all_samples = self.copy_training_data()
        x_train_all_samples = self.shuffle_data(x_train_all_samples)
        y_train_all_samples = self.shuffle_data(y_train_all_samples)
        num_validation_samples = int(self.validation_split * x_train_all_samples.shape[0])
        x_train = x_train_all_samples[:-num_validation_samples]
        y_train = y_train_all_samples[:-num_validation_samples]
        x_val = x_train_all_samples[-num_validation_samples:]
        y_val = y_train_all_samples[-num_validation_samples:]
        #convert y to 3d, for time distributed dense with sparse_categorical_crossentropy loss
        y_train = np.expand_dims(y_train,axis = 2)
        y_val = np.expand_dims(y_val,axis = 2)
        return(x_train,y_train,x_val,y_val)

    def fit_model(self):
        self.compile_model()                                                          
        x_train,y_train,x_val,y_val = self.create_data_validation_split()
        training_model_history = self.training_model.fit(
                                                    x_train, 
                                                    y_train,
                                                    batch_size = self.batch_size,
                                                    epochs = self.num_epochs,
                                                    validation_data = (x_val, y_val))
        return(training_model_history)

    def output_model_summary(self):
        print('model summary',self.training_model.summary())

    def output_model_layer_description(self):
        layer_names_list = [self.training_model.layers[x].name 
                          for x in range(len(self.training_model.layers))
                          ]   
        layer_descriptions_list = [[str(e) for e in self.training_model.layers[x].trainable_weights] 
                                 for x in range(len(self.training_model.layers))]   
        layer_description = list(zip(layer_names_list,layer_descriptions_list))
        print('Layer Descriptions',layer_description)

    @abstractmethod
    def predict_with_model(self,input_sentence):
        self.training_model.predict(input_sentence)


class LSTMModel(LanguageModel):                                    
    
    '''Class to build keras LSTM Model with word embeddings for text generation
    
    
    # Arguments:
        embed_dim: embedding dimension for word vectors 
        mask_zero: whether to mask a value (default 0) when training (T,F)
        trainable: whether embedding matrix should be trainingable (T,F)
        lstm_hidden_dim: dimension of hidden layer for LSTM unit
        return_sequences: whether to return the full sequences (T,F)
        activation: activation functio nto use in output layer   
    '''

    def __init__(self, *args, **kwargs):    
        self.embed_dim = kwargs.pop('embed_dim')
        self.mask_zero = kwargs.pop('mask_zero',True)
        self.trainable = kwargs.pop('trainable',False)
        self.lstm_hidden_dim = kwargs.pop('lstm_hidden_dim',128)
        self.return_sequences = kwargs.pop('return_sequences',True)
        self.activation = kwargs.pop('activation','softmax')
        self.use_pretrained_word_embeddings = kwargs.pop('use_pretrained_word_embeddings',False)
        self.embedding_matrix = kwargs.pop('embedding_matrix',None)
        super(LSTMModel, self).__init__(*args, **kwargs)

    def check_pretrained_word_embeddings_inputs_match(self):
        if self.use_pretrained_word_embeddings and self.embedding_matrix is None:
            raise ValueError(
                      'use_pretrained_word_embeddings is True, but no embedding_matrix'\
                    + 'has been provided to constructor')
        if not self.use_pretrained_word_embeddings and self.embedding_matrix is not None:
            raise ValueError(
                        'use_pretrained_word_embeddings is False, but an embedding_matrix'\
                      + 'has been provided to constructor')   
            
    def build_model_architecture(self):
        self.check_pretrained_word_embeddings_inputs_match()
        lstm_model = Sequential()
        if self.use_pretrained_word_embeddings and self.embedding_matrix is not None:
            lstm_model.add(
                    Embedding(
                            input_dim = self.vocab_size+1,
                            output_dim = self.embed_dim,
                            input_length = self.num_timesteps,
                            mask_zero = self.mask_zero,
                            weights = [self.embedding_matrix],
                            trainable = self.trainable))            
        if not self.use_pretrained_word_embeddings and self.embedding_matrix is None:
            lstm_model.add(
                    Embedding(
                            input_dim = self.vocab_size+1,
                            output_dim = self.embed_dim,
                            input_length = self.num_timesteps,
                            mask_zero = self.mask_zero,
                            trainable = self.trainable))           
        lstm_model.add(LSTM(self.lstm_hidden_dim,return_sequences = self.return_sequences))                                                                
        lstm_model.add(TimeDistributed(Dense(self.vocab_size+1,activation = self.activation)))                                                    
        return lstm_model

    def predict_with_model(self):
        print('in subclass predict method')


class WordEmbedder:
    
    '''Class to process word embeddings
    
    
    # Arguments:
        embedding_directory: directory where pretrained embeddings are located 
        embedding_file: name of embedding file
        vocab_size: vocab size for keras model
        embed_dim: dimension for word vectors
        sentence_start_token: token to indicate start of sentence
        sentence_end_token: token to indicate end of sentence
        unknown_token: token for unknown words
        word_to_index_dict: dict the maps words to numerical indices
    ''' 
    
    def __init__(
                self,
                embedding_directory,
                embedding_file,
                vocab_size,
                embed_dim,
                sentence_start_token,
                sentence_end_token,
                unknown_token,
                word_to_index_dict):
        
        self.embedding_directory = embedding_directory
        self.embedding_file = embedding_file
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.sentence_start_token = sentence_start_token
        self.sentence_end_token = sentence_end_token
        self.unknown_token = unknown_token
        self.word_to_index_dict = word_to_index_dict
            
    def create_word_to_embeddings_dict(self):
        word_to_embeddings_dict = {}
        file_embed = open(os.path.join(self.embedding_directory,self.embedding_file))
        for line in file_embed:
            computed_embeddings = line.split()
            words = computed_embeddings[0]
            word_vectors = np.asarray(computed_embeddings[1:],dtype = 'float32')
            word_to_embeddings_dict[words] = word_vectors
        file_embed.close()
        return word_to_embeddings_dict
    
    def check_word_embedding_dimension_correct(self,word_to_test):
        word_to_embeddings_dict = self.create_word_to_embeddings_dict()
        test_vec = word_to_embeddings_dict[word_to_test]
        if test_vec.shape[0] != self.embed_dim:
            raise ValueError('Embedding vectors do not have the correct embedding dimension!')
    
    def create_init_vectors_for_tokens(self,random_seed_input):
        np.random.seed(random_seed_input)
        init_embedding_vec = np.random.rand(1,self.embed_dim)
        return(init_embedding_vec)
    
    def create_embedding_matrix(self):
        num_words_to_embed = self.vocab_size+1
        embedding_matrix = np.zeros((num_words_to_embed, self.embed_dim))
    
        init_embedding_vec_for_start_token = self.create_init_vectors_for_tokens(121)
        init_embedding_vec_for_end_token = self.create_init_vectors_for_tokens(243)
        init_embedding_vec_for_unknown_token = self.create_init_vectors_for_tokens(377)
        
        embeddings_to_index_dict = self.create_word_to_embeddings_dict()
        self.check_word_embedding_dimension_correct('the')
        self.check_word_embedding_dimension_correct('a')        
        for word, i in self.word_to_index_dict.items():
            embedding_vector = embeddings_to_index_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            if embedding_vector is None and word==self.unknown_token:               
                embedding_matrix[i] = init_embedding_vec_for_unknown_token
            if embedding_vector is None and word  ==  self.sentence_start_token:
                embedding_matrix[i] = init_embedding_vec_for_start_token
            if embedding_vector is None and word==self.sentence_end_token:
                embedding_matrix[i] = init_embedding_vec_for_end_token    
        return(embedding_matrix)        
