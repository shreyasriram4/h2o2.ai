import pandas as pd
import numpy as np
import random
import math
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import auc
import nltk
from nltk.tokenize import word_tokenize
import joblib
import gensim
from gensim.models import Word2Vec
from src.models.classifier import Classifier
from src.utils.file_util import FileUtil
import keras
from keras.preprocessing.text import one_hot, Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Embedding, Input, LSTM, ReLU, Dropout, Bidirectional
import tensorflow
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


class LSTM(Classifier):
    def __init__(self, load_model=False):
        self.load_model = load_model
        self.saved_model_path = FileUtil().LSTM_SENTIMENT_MODEL_DIR
        self.LSTM_cofig = FileUtil.get_config()["LSTM"]
        self.batch_size = self.LSTM_config["batch_size"]
        self.target_col = self.LSTM_config["target_col"]
        self.text_col = self.LSTM_config["text_col"]
        self.vector_size = self.LSTM_config["vector_size"]
        self.window = self.LSTM_config["window"]
        self.min_count = self.LSTM_config["min_count"]
        self.sg = self.LSTM_config["sg"]
        self.verbose = self.LSTM_config["verbose"]
        self.patience = self.LSTM_config["patience"]
        self.min_delta = self.LSTM_config["min_delta"]
        self.epochs = self.LSTM_config["epochs"]
        self.lr = self.LSTM_config["learning rate"]
        self.input_length = self.LSTM_config["input_length"]
        self.history_embedding = ""
        self.callbacks_list = []
        self.word_set = set()
        self.w2v_model = ""
        self.model = ""
        self.tokenizer = Tokenizer()
        self.embed_matrix = ""
        if load_model:
            if not FileUtil.check_dir_exists(self.saved_model_path):
                raise FileNotFoundError("There is no saved model in path",
                                        self.saved_model_path)
            self.model = load_model(self.saved_model_path)

    def tokenize(self, df):
        nltk.download('punkt')
        df['cleaned_text_new'] = df['cleaned_text'].apply(
            lambda x: word_tokenize(x))

        return df

    def train_w2v_model(self, df):

        X = df[self.text_col]
        w2v_model = gensim.models.Word2Vec(X,
                                           self.vector_size,
                                           self.window,
                                           self.min_count,
                                           self.sg)
        self.w2v_model = w2v_model

        self.tokenizer.fit_on_texts(df['cleaned_text'])
        # number of unique text in the data
        vocab_size = len(self.tokenizer.word_index) + 1
        self.vocab_size = vocab_size

        return w2v_model

    def get_word_vectors(self, df):

        # this converts texts into some numeric sequences
        encd_rev = self.tokenizer.texts_to_sequences(df['cleaned_text'])

        # we dont pad to a maximum length of 925 but use 50 if not the train
        # data will be very sparse
        pad_rev = pad_sequences(encd_rev, maxlen=50, padding='post')

        return pad_rev

    def get_embedding_matrix(self):

        embed_matrix = np.zeros(shape=(self.vocab_size, self.vector_size))
        for word, i in self.tokenizer.word_index.items():

            # check if word is in the vocabulary learned by the w2v model
            if word in self.w2v_model.wv.index_to_key:
                # get the word vector from w2v model
                embed_matrix[i] = self.w2v_model.wv[word]
                # else the embed_vector corressponding
                # to that vector will stay zero.
        self.embed_matrix = embed_matrix
        return embed_matrix

    def build_model(self):
        # Build the LSTM Model
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.vector_size,
                            input_length=self.input_length, weights=[self.embed_matrix]))
        # embeddings_initializer = Constant(embed_matrix)))
        model.add(LSTM(128))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.20))
        model.add(Dense(2, activation='sigmoid'))

        model.compile(optimizer=keras.optimizers.RMSprop(
            learning_rate=self.lr), loss='binary_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(
            self.saved_model_path, monitor="val_accuracy", verbose=1,
            save_best_only=True, mode="max"
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=self.patience,
            verbose=self.verbose, min_delta=self.min_delta
        )

        self.callbacks_list = [checkpoint, early_stopping]

        self.model = model

        return model

    def fit(self, train, val):
        assert self.load_model is not True

        X_train_vect = self.get_word_vectors(train)
        X_val_vect = self.get_word_vectors(val)

        y_train = keras.utils.to_categorical(train[self.target_col])
        y_val = keras.utils.to_categorical(val[self.target_col])
        history_embedding = self.model.fit(
            X_train_vect, y_train, epochs=self.epochs,
            batch_size=self.batch_size_lstm, validation_data=(
                X_val_vect, y_val),
            callbacks=self.callbacks_list)
        self.history_embedding = history_embedding
        return history_embedding

    def plot_training_acc(self):

        accs = self.history_embedding.history['accuracy']
        val_accs = self.history_embedding.history['val_accuracy']

        plt.plot(accs, c='b', label='train accuracy')
        plt.plot(val_accs, c='r', label='validation accuracy')
        plt.legend(loc='upper right')
        plt.show()

        plt.savefig(FileUtil().LSTM_TRAINING_ACC_GRAPH_FILE_PATH)

    def plot_training_loss(self):

        losses = self.history_embedding.history['loss']
        val_losses = self.history_embedding.history['val_loss']

        plt.plot(losses, c='b', label='train loss')
        plt.plot(val_losses, c='r', label='validation loss')
        plt.legend(loc='upper right')
        plt.show()

        plt.savefig(FileUtil().LSTM_TRAINING_LOSSES_GRAPH_FILE_PATH)

    def predict(self, valid):

        self.model = load_model(self.saved_model_path)
        X_val_vect = self.get_word_vectors(valid)
        # Use the trained model to make predictions on the val data

        LSTM_y_pred = self.model.predict(X_val_vect)
        LSTM_test_pred = []
        for pred in LSTM_y_pred:
            if pred[0] >= 0.5:
                LSTM_test_pred.append(0)
            else:
                LSTM_test_pred.append(1)

        return LSTM_test_pred, LSTM_y_pred

    def evaluate(self, valid):
        LSTM_test_pred, y_pred = self.predict(valid)
        y_test = keras.utils.to_categorical(valid[self.target_col])

        # get actual test response
        test_actual = []

        for pred in y_test:
            if pred[0] >= 0.5:
                test_actual.append(0)
            else:
                test_actual.append(1)

        # keep probabilities for the positive outcome only
        LSTM_y_probs = y_pred[:, 1]

        LSTM_precision, LSTM_recall, LSTM_thresholds = precision_recall_curve(
            test_actual, LSTM_y_probs)

        ap = average_precision_score(test_actual,  LSTM_y_probs)
        pr_auc = auc(LSTM_recall, LSTM_precision)

        return ap, pr_auc
