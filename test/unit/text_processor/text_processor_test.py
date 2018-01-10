#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import nose
import text_processor.text_processor as textp


class TestTextListCleaner(unittest.TestCase):
    
    def setUp(self):
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"
        self.string1='\n\n\n\t\t\tCALL FOR PRESENTATIONS\n\t\n  NAVY SCIENce' \
                 + 'VISUALIZATION VIRTUAL REALITY SEMINAR\n\n\t\t\tTuesday,' \
                 + 'June 22, 1993\n\n\t    Carderock Division, Naval Surface' \
                 + 'Warfare Center\n\t    (formerly the David Taylor Research' \
                 + 'Center)\n\t\t\t  Bethesda, Maryland\n\nSP.'
        
        self.string2 = 'Although I think that it\'s inevitable that the team\n'\
                   + 'will improve with a player such as Lemieux or Gretzky,'\
                   +  'simply because they\nmake people around them better.'\
                   +  '\n\n>But winning sure as hell helps.  ;-)'  
                          
        self.text_list = [self.string1,self.string2]
        self.text_list_processor_obj = textp.TextListCleaner(self.text_list)
    
    def test_text_list_processor_returns_constructor_params(self):
        self.assertTrue(self.string1 in self.text_list_processor_obj.text_list)
        self.assertTrue(self.string2 in self.text_list_processor_obj.text_list)

    def test_copy_list(self):
        text_list_copy = self.text_list_processor_obj.copy_list()        
        self.assertNotEqual(id(text_list_copy),id(self.text_list_processor_obj.text_list))
        self.assertCountEqual(text_list_copy,self.text_list_processor_obj.text_list)
        
    def test_lowercase_text(self):
        text_list_lowercase = self.text_list_processor_obj.lowercase_text(self.text_list) 
        self.assertFalse(set(text_list_lowercase) & set(self.text_list_processor_obj.text_list))
        string1_lowercase = '\n\n\n\t\t\tcall for presentations\n\t\n  navy science' \
                 + 'visualization virtual reality seminar\n\n\t\t\ttuesday,' \
                 + 'june 22, 1993\n\n\t    carderock division, naval surface' \
                 + 'warfare center\n\t    (formerly the david taylor research' \
                 + 'center)\n\t\t\t  bethesda, maryland\n\nsp.'
                 
        string2_lowercase = 'although i think that it\'s inevitable that the team\n'\
                   + 'will improve with a player such as lemieux or gretzky,'\
                   + 'simply because they\nmake people around them better.'\
                   + '\n\n>but winning sure as hell helps.  ;-)'         
                 
        self.assertTrue(string1_lowercase in text_list_lowercase)                 
        self.assertTrue(string2_lowercase in text_list_lowercase)
                
    def test_remove_punctuation_from_text(self):
        text_list_no_punct = self.text_list_processor_obj.remove_punctuation_from_text(
                                                                                self.text_list,'.')
        self.assertFalse(set(text_list_no_punct) & set(self.text_list_processor_obj.text_list))        
        string1_no_punct = '\n\n\n\t\t\tCALL FOR PRESENTATIONS\n\t\n  NAVY SCIENce' \
                 + 'VISUALIZATION VIRTUAL REALITY SEMINAR\n\n\t\t\tTuesday' \
                 + 'June 22 1993\n\n\t    Carderock Division Naval Surface' \
                 + 'Warfare Center\n\t    formerly the David Taylor Research' \
                 + 'Center\n\t\t\t  Bethesda Maryland\n\nSP.'
                 
        print(text_list_no_punct)         
        self.assertTrue(string1_no_punct in text_list_no_punct)  

    def test_remove_tab_from_text(self):
        test_string_list = [
                'Hello\t this is \t\t the news\n.',
                'I thought \t it was won\tderful.'
                ]
        test_string_no_tab = self.text_list_processor_obj.remove_tab_from_text(test_string_list)
        self.assertEqual(
                test_string_no_tab,
                [
                 'Hello this is  the news\n.',
                 'I thought  it was wonderful.'
                ])               
        
    def test_remove_newline_from_text(self):
        test_string_list = [
                'Hello\t this is \t\t the news\n.',
                'I thought \t it was won\tderful.'
                ]
        test_string_no_newline = self.text_list_processor_obj.\
                                    remove_newline_from_text(test_string_list)
        self.assertEqual(
                test_string_no_newline,
                [
                 'Hello\t this is \t\t the news.',
                 'I thought \t it was won\tderful.'
                 ])                  

    def test_remove_extra_spaces_from_text(self):
        test_string_list = [
                        'this   has many spaces.',
                        'also this   has   too many.'
                         ]
        test_string_no_extra_spaces = self.text_list_processor_obj.\
                                remove_extra_spaces_from_text(test_string_list)
        self.assertEqual(
                test_string_no_extra_spaces,
                [
                'this has many spaces.',
                'also this has too many.'
                ])

    def test_clean_text(self):
        clean_text_list = self.text_list_processor_obj.clean_text()
        string1_clean='call for presentations navy sciencevisualization '\
                      + 'virtual reality seminartuesdayjune 22 1993 carderock '\
                      + 'division naval surfacewarfare center formerly the '\
                      + 'david taylor researchcenter bethesda marylandsp.'
        self.assertTrue(string1_clean in clean_text_list)
        
    def test_tokenize_clean_string_into_sentences(self):
        sentence_list = self.text_list_processor_obj.tokenize_clean_string_into_sentences()
        string_sentence_tokenized = 'call for presentations navy sciencevisualization '\
                      + 'virtual reality seminartuesdayjune 22 1993 carderock '\
                      + 'division naval surfacewarfare center formerly the '\
                      + 'david taylor researchcenter bethesda marylandsp.'
        self.assertTrue(string_sentence_tokenized in sentence_list)
        
        
class TestKerasTextListPreparer(unittest.TestCase):
    
        def setUp(self):
            self.string1 = 'The book could have provided an \n\t'\
                          + 'assessment of contemporary cultures.'
        
            self.string2 = 'It was said the governor ;! could have\n\t'\
                          + ' stopped an outage.'
                          
            self.text_list= [self.string1,self.string2]
            self.keras_text_list_preparer_obj = textp.KerasTextListPreparer(
                    self.text_list,'SENTENCE_START','SENTENCE_END','UNKNOWN_TOKEN',10,12)
        
        def test_keras_text_list_preparer_returns_constructor_params(self):
            self.assertTrue(
                    self.string1 in self.keras_text_list_preparer_obj.text_cleaner.text_list)
            self.assertTrue(
                    self.string2 in self.keras_text_list_preparer_obj.text_cleaner.text_list)
            self.assertEqual(
                    self.keras_text_list_preparer_obj.vocab_size,10)
            self.assertEqual(
                    self.keras_text_list_preparer_obj.sentence_start_token,'SENTENCE_START')
            self.assertEqual(
                    self.keras_text_list_preparer_obj.sentence_end_token,'SENTENCE_END')
            self.assertEqual(
                    self.keras_text_list_preparer_obj.unknown_token,'UNKNOWN_TOKEN')
            self.assertCountEqual(
                    self.keras_text_list_preparer_obj.sentence_list,
                    ['SENTENCE_START the book could have provided an '\
                     + 'assessment of contemporary cultures'\
                     +' SENTENCE_END',
                     'SENTENCE_START it was said the governor could have'\
                     + ' stopped an outage SENTENCE_END'
                     ])
            self.assertCountEqual(
                    self.keras_text_list_preparer_obj.tokenize_sentences,
                    [[
                    'SENTENCE_START', 'the', 'book', 'could', 'have', 
                    'provided', 'an', 'assessment', 'of', 'contemporary',
                    'cultures', 'SENTENCE_END'], 
                     [
                     'SENTENCE_START', 'it', 'was', 'said', 'the', 'governor',
                     'could', 'have', 'stopped', 'an', 'outage', 
                     'SENTENCE_END'
                     ]])

        def test_create_word_to_index_dict(self):
            keras_text_list_preparer_obj = textp.KerasTextListPreparer(
                    self.text_list,'SENTENCE_START','SENTENCE_END','UNKNOWN_TOKEN',10,12)
            keras_text_list_preparer_obj.vocab = [
                                   ('an', 2), ('SENTENCE_START', 2),
                                   ('the', 2), ('SENTENCE_END', 2), 
                                   ('could', 2), ('have', 2),
                                   ('stopped', 1), ('governor', 1),
                                   ('assessment', 1)
                                   ]
            word_to_index_dict = keras_text_list_preparer_obj.create_word_to_index_dict()
            self.assertCountEqual(
                            word_to_index_dict,
                            {'an': 1, 'SENTENCE_START': 2, 'stopped': 7, 
                              'the': 3, 'assessment': 9, 'SENTENCE_END': 4, 
                              'UNKNOWN_TOKEN': 10, 'governor': 8, 'could': 5,
                              'have': 6
                              })

        def test_create_index_to_word_dict(self):
            keras_text_list_preparer_obj = textp.KerasTextListPreparer(
                    self.text_list,'SENTENCE_START','SENTENCE_END','UNKNOWN_TOKEN',10,12)
            keras_text_list_preparer_obj.vocab = [
                                   ('an', 2), ('SENTENCE_START', 2),
                                   ('the', 2), ('SENTENCE_END', 2), 
                                   ('could', 2), ('have', 2),
                                   ('stopped', 1), ('governor', 1),
                                   ('assessment', 1)
                                   ]
            keras_text_list_preparer_obj.word_to_index_dict = keras_text_list_preparer_obj.\
                                                            create_word_to_index_dict()
            index_to_word_dict = keras_text_list_preparer_obj.create_index_to_word_dict()
            self.assertCountEqual(
                             index_to_word_dict,
                             {0: 'zero padding', 1: 'an', 2: 'SENTENCE_START',
                              3: 'the', 4: 'SENTENCE_END', 5: 'could',
                              6: 'have', 7: 'stopped', 8: 'governor', 
                              9: 'assessment', 10: 'UNKNOWN_TOKEN'
                              })
    
        def test_calculate_num_unique_words_in_corpus(self):
            num_words = self.keras_text_list_preparer_obj.calculate_num_unique_words_in_corpus()
            self.assertEqual(num_words,18)

        def test_check_word_to_index_value(self):
            word_to_index_dict_test = self.keras_text_list_preparer_obj.create_word_to_index_dict()
            word_to_index_dict_test['wrong_value'] = 0
            self.assertRaises(
                    ValueError,
                    self.keras_text_list_preparer_obj.check_word_to_index_value,
                    word_to_index_dict_test)
                               
        def test_create_training_sentences(self):            
            keras_text_list_preparer_obj = textp.KerasTextListPreparer(
                    self.text_list,'SENTENCE_START','SENTENCE_END','UNKNOWN_TOKEN',10,12)
            keras_text_list_preparer_obj.tokenize_sentences = [
                    ['SENTENCE_START', 'the', 'book', 'could', 'have', 
                     'provided', 'an', 'assessment', 'of', 'contemporary',
                     'cultures', 'SENTENCE_END'
                     ], 
                    ['SENTENCE_START', 'it', 'was', 'said', 'the', 'governor',
                     'could', 'have', 'stopped', 'an', 'outage', 
                     'SENTENCE_END'
                     ]]
            keras_text_list_preparer_obj.word_to_index_dict = {
                              'an': 1, 'SENTENCE_START': 2, 'stopped': 7, 
                              'the': 3, 'assessment': 9, 'SENTENCE_END': 4, 
                              'UNKNOWN_TOKEN': 10, 'governor': 8, 'could': 5,
                              'have': 6
                              }
            training_sentences = keras_text_list_preparer_obj.create_training_sentences()                       
            self.assertEqual(
                             training_sentences,
                             [['SENTENCE_START', 'the', 'UNKNOWN_TOKEN',
                               'could', 'have', 'UNKNOWN_TOKEN', 'an', 
                               'assessment', 'UNKNOWN_TOKEN', 'UNKNOWN_TOKEN',
                               'UNKNOWN_TOKEN', 'SENTENCE_END'
                               ],
                            ['SENTENCE_START', 'UNKNOWN_TOKEN', 
                             'UNKNOWN_TOKEN', 'UNKNOWN_TOKEN', 'the', 
                             'governor', 'could', 'have', 'stopped', 'an', 
                             'UNKNOWN_TOKEN', 'SENTENCE_END'
                             ]])  
                       
        def test_create_training_data(self):
            keras_text_list_preparer_obj = textp.KerasTextListPreparer(
                    self.text_list,'SENTENCE_START','SENTENCE_END','UNKNOWN_TOKEN',10,12)
            keras_text_list_preparer_obj.word_to_index_dict={
                              'an': 1, 'SENTENCE_START': 2, 'stopped': 7, 
                              'the': 3, 'assessment': 9, 'SENTENCE_END': 4, 
                              'UNKNOWN_TOKEN': 10, 'governor': 8, 'could': 5,
                              'have': 6
                              }
            keras_text_list_preparer_obj.tokenize_sentences = [
                    ['SENTENCE_START', 'the', 'book', 'could', 'have', 
                     'provided', 'an', 'assessment', 'of', 'contemporary',
                     'cultures', 'SENTENCE_END'
                     ], 
                    ['SENTENCE_START', 'it', 'was', 'said', 'the', 'governor',
                     'could', 'have', 'stopped', 'an', 'outage', 
                     'SENTENCE_END'
                     ]]
            
            x_training_data,y_training_data = keras_text_list_preparer_obj.create_training_data()            
            self.assertEqual(keras_text_list_preparer_obj.word_to_index_dict['SENTENCE_START'],2) 
            self.assertEqual(keras_text_list_preparer_obj.word_to_index_dict['SENTENCE_END'],4)
            self.assertCountEqual(x_training_data[0],[2, 3, 10, 5, 6, 10, 1, 9, 10, 10, 10])
            self.assertCountEqual(x_training_data[1], [2, 10, 10, 10, 3, 8, 5, 6, 7, 1, 10])              
            self.assertCountEqual(y_training_data[0],[3, 10, 5, 6, 10, 1, 9, 10, 10, 10, 4])
            self.assertCountEqual(y_training_data[1], [10, 10, 10, 3, 8, 5, 6, 7, 1, 10, 4]) 

        def test_create_padded_training_data(self):            
            keras_text_list_preparer_obj = textp.KerasTextListPreparer(
                    self.text_list,'SENTENCE_START','SENTENCE_END','UNKNOWN_TOKEN',10,12)
            keras_text_list_preparer_obj.word_to_index_dict = {
                              'an': 1, 'SENTENCE_START': 2, 'stopped': 7, 
                              'the': 3, 'assessment': 9, 'SENTENCE_END': 4, 
                              'UNKNOWN_TOKEN': 10, 'governor': 8, 'could': 5,
                              'have': 6
                              }
            keras_text_list_preparer_obj.tokenize_sentences = [
                    ['SENTENCE_START', 'the', 'book', 'could', 'have', 
                     'provided', 'an', 'assessment', 'of', 'contemporary',
                     'cultures', 'SENTENCE_END'
                     ], 
                    ['SENTENCE_START', 'it', 'was', 'said', 'the', 'governor',
                     'could', 'have', 'stopped', 'an', 'outage', 
                     'SENTENCE_END'
                     ]]
            x_pad_training_data,y_pad_training_data=keras_text_list_preparer_obj.\
                                               create_padded_training_data()
            self.assertCountEqual(
                                  x_pad_training_data[0],
                                  [2, 3, 10, 5, 6, 10, 1, 9, 10, 10, 10, 0])
            self.assertCountEqual(
                                  x_pad_training_data[1],
                                  [2, 10, 10, 10, 3, 8, 5, 6, 7, 1, 10, 0])  
            self.assertCountEqual(
                                  y_pad_training_data[0],
                                  [3, 10, 5, 6, 10, 1, 9, 10, 10, 10, 4, 0])
            self.assertCountEqual(
                                  y_pad_training_data[1],
                                  [10, 10, 10, 3, 8, 5, 6, 7, 1, 10, 4, 0]) 


if __name__=='__main__':
    nose.run(defaultTest=__name__)
