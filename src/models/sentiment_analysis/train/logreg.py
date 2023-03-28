import pandas as pd
import numpy as np
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, precision_recall_curve, average_precision_score, auc
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import gensim
from gensim.models import Word2Vec
from src.models.classifier import Classifier
from src.utils.file_util import FileUtil


class LOGREG(Classifier):
    def __init__(self, load_model=False):
        self.load_model = load_model
        self.saved_model_path = FileUtil.BERT_SENTIMENT_MODEL_DIR
        self.logreg_config = FileUtil.get_config()["LOGREG"]
        self.batch_size = self.logreg_config["batch_size"]
        self.target_col = self.logreg_config["target_col"]
        self.text_col = self.logreg_config["text_col"]
        self.vector_size = self.logreg_config["vector_size"]
        self.window = self.logreg_config["window"]
        self.min_count = self.logreg_config["min_count"]
        self.sg = self.logreg_config["sg"]

    def tokenize(self, df):
        nltk.download('punkt')
        df['cleaned_text_new'] = df['cleaned_text'].apply(
            lambda x: word_tokenize(x))

        return df
            
    def train_w2v_model(self, train):

        X_train = train[self.text_col]
        w2v_model = gensim.models.Word2Vec(X_train,
                                           self.vector_size,
                                           self.window,
                                           self.min_count,
                                           self.sg)

        return w2v_model
    
    def get_word_vectors(self, train, valid):

        w2v_model = self.train_w2v_model(train,
                                         self.vector_size,
                                         self.window,
                                         self.min_count,
                                         self.sg)

        X_train = train[self.text_col]

        X_valid = valid[self.text_col]

        # words that appear in the train w2v model
        words = set(w2v_model.wv.index_to_key)

        # train data
        X_train_vect = np.array([np.array(
            [w2v_model.wv[i] for i in ls if i in words])
            for ls in X_train])
        
        # test data
        X_valid_vect = np.array([np.array(
            [w2v_model.wv[i] for i in ls if i in words])
            for ls in X_valid])
        
        # Maintain consistency in each sentence vector so
        # that the size of our matrix is consistent

        # Compute sentence vectors by averaging the word vectors 
        # for the words contained in the sentence

        X_train_vect_avg = []
        for v in X_train_vect:
            if v.size:
                # take average weights across all the word 
                # vectors within the sentence vector
                X_train_vect_avg.append(v.mean(axis=0))
            else:
                # else set zero vector of size 100 because the 
                # size of vector that we initially set is 100
                X_train_vect_avg.append(np.zeros(self.vector_size,
                                                 dtype=float))
                
        X_valid_vect_avg = []
        for v in X_valid_vect:
            if v.size:
                # take average weights across all the word 
                # vectors within the sentence vector
                X_valid_vect_avg.append(v.mean(axis=0))
            else:
                # else set zero vector of size 100 because the 
                # size of vector that we initially set is 100
                X_valid_vect_avg.append(np.zeros(self.vector_size,
                                                 dtype=float))

        return X_train_vect_avg, X_valid_vect_avg

    def fit(self, train, valid):
        assert self.load_model is not True

        X_train_vect_avg = self.get_word_vectors(train, valid)[0]

        y_train = train[self.target_col]
        logr = LogisticRegression(random_state=1)
        logr.fit(X_train_vect_avg, y_train.values.ravel())

        return logr

    def predict(self, train, valid):

        logreg_model = self.fit(train, valid)

        X_valid_vect_avg = self.get_word_vectors(train, valid)[1]

        # Use the trained model to make predictions on the val data
        y_pred = logreg_model.predict(X_valid_vect_avg)
        y_test = valid[self.target_col]
        LR_y_pred = logreg_model.predict_proba(X_valid_vect_avg)
        lr_y_test = y_test.copy()

        return LR_y_pred, lr_y_test
    
    def evaluate(self, train, valid):
        y_pred, y_label = self.predict(train, valid)

        # keep probabilities for the positive outcome only
        LR_y_probs = y_pred[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(
            y_label, LR_y_probs
            )
       
        ap = average_precision_score(y_label,  LR_y_probs)
        pr_auc = auc(recall, precision)

        return ap, pr_auc


    