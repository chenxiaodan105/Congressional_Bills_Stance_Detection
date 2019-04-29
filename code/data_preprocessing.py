#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:14:36 2019

@author: xiaodanchen
"""
import pandas as pd
import numpy as np
import pickle
import string
import re
import numpy as np
import scipy
import os
import collections
from tqdm import tqdm_notebook as tqdm
import warnings
warnings.filterwarnings('ignore')

#sklearn
from sklearn.model_selection import train_test_split

#nltk for nlp
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

#relabel model
from relabel_model import RuleBasedStanceDetection

def read_data(data_path):
    '''
    read csv file
    input: your csv file location 
    output: DataFrame
    '''
    data = pd.read_csv(data_path,index_col=0)
    return data

def save_jpeg(path,name):
    fn = '%s.jpeg'%name
    plt.savefig(os.path.join(path,fn))
    
def tokenize_text(text):
    '''
    tokenization
    '''
    tokens = word_tokenize(text.lower())
    tokens = [token.strip() for token in tokens]
    return tokens

def remove_stopwords(text):
    '''
    remove stop words
    '''
    stop = list(set(stopwords.words('english')))
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stop]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_special_characters(text):
    '''
    remove special characters: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    '''
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_non_alphabetic_characters(text):
    '''
    remove non-alphabetic characters and numbers
    '''
    tokens = tokenize_text(text)
    tokens = [w for w in tokens if w.isalpha()]
    return ' '.join(tokens)

def remove_tokens_with_length(text,length):
    '''
    remove tokens with length less than or equal the input length
    '''
    tokens = tokenize_text(text)
    tokens = [w for w in tokens if len(w)>length]
    return ' '.join(tokens)
    
def text_preprocessing(corpus):
    '''
    preprocess 
    '''
    new_corpus = []
    for text in tqdm(corpus):
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        #text = remove_non_alphabetic_characters(text)
        #text = remove_tokens_with_length(text,1)
        new_corpus.append(text)
    return new_corpus

def relabel_data(data):
    sd = RuleBasedStanceDetection()
    X,y = data['text'].values, data['tagged'].values
    X = text_preprocessing(X)
    for i in tqdm(range(len(X))):
        if y[i] == -1:
            y[i] = sd.stance_detection_labeler(X[i])
    return X,y

def split(data):
    X,y = relabel_data(data)
    train_corpus, test_corpus, train_labels, test_labels = train_test_split(X,y,test_size=0.3)
    return train_corpus, test_corpus, train_labels, test_labels

# relabelling class using regular expression
class RuleBasedStanceDetection(object):
    def __init__(self):
        self.positive_detector = re.compile('support')
        self.negative_detector = re.compile('opposition|oppose')
        self.confused_detector = re.compile('passed|introduce|introduction|rise|will vote')
        self.contain_bill = re.compile('H[0-9]{4}|H.R.|[A,a]ct')
        self.pn_count = []

    def stance_detector(self, speech, pn_ratio=1, cutoff=None):
        """
        predict by the count of positive/negative keyword
        Args:
            speech(str): the speech
            pn_ratio(int,float): the parameter to control the weight of negative keywords
            when both positive and negative keywords appear in the speech
            cutoff(int): the cutoff point of the speech, only detect the keyword before cutoff
        Returns:
            int: -1(negative),1(positive),0(not detected),2(confused),3(contain bill)
        """
        if cutoff:
            speech = speech[:cutoff]
        if self.positive_detector.search(speech):
            positive_count = len(self.positive_detector.findall(speech))
        else:
            positive_count = 0
        if self.negative_detector.search(speech):
            negative_count = len(self.negative_detector.findall(speech))
        else:
            negative_count = 0
        if positive_count == 0 and negative_count == 0:
            if self.confused_detector.search(speech):
                return 2
            if self.contain_bill.search(speech):
                return 3
            return 0
        else:
            self.pn_count.append([positive_count, negative_count])
            # print(positive_count, negative_count)
            if positive_count > pn_ratio * negative_count:
                return 1
            else:
                return -1

    def stance_detection_labeler(self, speech, strict=True, pn_ratio=1, cutoff=None):
        stance = self.stance_detector(speech, pn_ratio, cutoff)
        if strict:  # only relabel when certain
            if stance == 1 or stance == -1:
                return 1
            else:
                return -1
        else:  # relabel when possible
            if stance != 0:
                return 1
            else:
                return -1

    def stance_classification_labeler(self, speech, pn_ratio=1, cutoff=None):
        stance = self.stance_detector(speech, pn_ratio, cutoff)
        if stance == 1 or stance == -1:
            return stance
        else:
            return 0
        
