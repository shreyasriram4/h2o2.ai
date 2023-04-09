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
from keras.layers import Dense, Flatten, Embedding, Input, LSTM, ReLU, Dropout
from keras.layers import Bidirectional
import tensorflow
from keras.utils import np_utils
from tensorflow.keras import optimizers
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


class Lstm(Classifier):
    """Lstm sentiment analysis class."""

    def __init__(self, load_model_bool=False):
        """
        Constructor for Lstm class.

        Args:
          load_model (bool): boolean value to indicate
          whether to load trained model or not

        Raises:
          FileNotFoundError: If load_model is True, but
          there is no trained model in the Lstm sentiment
          model directory
        """
        self.load_model = load_model_bool
        self.saved_model_path = FileUtil().LSTM_SENTIMENT_MODEL_DIR
        self.LSTM_config = FileUtil.get_config()["LSTM"]
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
        self.lr = self.LSTM_config["learning_rate"]
        self.input_length = self.LSTM_config["input_length"]
        self.history_embedding = ""
        self.callbacks_list = []
        self.word_set = set()
        self.w2v_model = ""
        self.model = ""
        self.tokenizer = Tokenizer()
        self.embed_matrix = ""
        self.vocab_size = 0
        if self.load_model:
            if not FileUtil.check_dir_exists(self.saved_model_path):
                raise FileNotFoundError("There is no saved model in path",
                                        self.saved_model_path)
            self.model = load_model(self.saved_model_path)

    def tokenize(self, df):
        """
        Tokenize data.

        Args:
          df (pd.DataFrame): dataframe 

        Returns:
          df (pd.DataFrame): dataframe with tokenized column
        """
        nltk.download('punkt')
        df['cleaned_text_new'] = df['cleaned_text'].apply(
            lambda x: word_tokenize(x))
        self.tokenizer.fit_on_texts(df['cleaned_text'])
        # number of unique text in the data
        vocab_size = len(self.tokenizer.word_index) + 1
        self.vocab_size = vocab_size
        return df

    def train_w2v_model(self, df):
        """
        Trains word to vector model.

        Args:
          df (pd.DataFrame): dataframe 

        Returns:
          w2v_model: Trained word to vector model
        """

        X = df[self.text_col]
        w2v_model = gensim.models.Word2Vec(X,
                                           vector_size=self.vector_size,
                                           window=self.window,
                                           min_count=self.min_count,
                                           sg=self.sg)
        self.w2v_model = w2v_model

        return w2v_model

    def get_word_vectors(self, df):
        """
        Converts texts to numeric sequences.

        Args:
          df (pd.DataFrame): dataframe

        Returns:
          pad_rev: Numeric representation of texts
        """

        # this converts texts into some numeric sequences
        encd_rev = self.tokenizer.texts_to_sequences(df['cleaned_text'])

        # we dont pad to a maximum length of 925 but use 50 if not the train
        # data will be very sparse
        pad_rev = pad_sequences(encd_rev, maxlen=50, padding='post')

        return pad_rev

    def get_embedding_matrix(self):
        """
        Get embedded matrix for the shifting of weights.

        Returns:
          embed_matrix: Embedded matrix
        """

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
        """
        Builds the lstm model with its layers
        based on the config file.

        Returns:
          model: Built and configured model
        """
        # Build the LSTM Model
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size,
                            output_dim=self.vector_size,
                            input_length=self.input_length,
                            weights=[self.embed_matrix]))
        # embeddings_initializer = Constant(embed_matrix)))
        model.add(LSTM(128))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.20))
        model.add(Dense(2, activation='sigmoid'))

        model.compile(optimizer=tensorflow.keras.optimizers.RMSprop(
            learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy'])

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
        """
        Fit Lstm model on the train data and val data

        Args:
          train (pd.DataFrame): train dataframe
          valid (pd.DataFrame): valid dataframe

        Returns:
          history_embedding: training history
        """
        assert self.load_model is not True

        X_train_vect = self.get_word_vectors(train)
        X_val_vect = self.get_word_vectors(val)

        y_train = keras.utils.np_utils.to_categorical(train[self.target_col])
        y_val = keras.utils.np_utils.to_categorical(val[self.target_col])
        history_embedding = self.model.fit(
            X_train_vect, y_train, epochs=self.epochs,
            batch_size=self.batch_size, validation_data=(
                X_val_vect, y_val),
            callbacks=self.callbacks_list)
        self.history_embedding = history_embedding
        return history_embedding

    def plot_training_metrics(self):
        """
        Plot and save LSTM training graph.

        """

        accs = self.history_embedding.history['accuracy']
        val_accs = self.history_embedding.history['val_accuracy']

        losses = self.history_embedding.history['loss']
        val_losses = self.history_embedding.history['val_loss']
        epochs = len(losses)

        plt.figure(figsize=(12, 4))
        for i, metrics in enumerate(zip([losses, accs], [val_losses, val_accs],
                                        ['Loss', 'Accuracy'])):
            plt.subplot(1, 2, i + 1)
            plt.plot(range(1, epochs + 1), metrics[0],
                     label='Training {}'.format(metrics[2]))
            plt.title("LSTM Training Graph")
            plt.plot(range(1, epochs + 1), metrics[1],
                     label='Validation {}'.format(metrics[2]))
            plt.title("LSTM Training Graph")
            plt.legend()
        FileUtil.create_dir_if_not_exists(
            FileUtil().SENTIMENT_ANALYSIS_EVAL_DIR)
        plt.savefig(FileUtil().LSTM_TRAINING_GRAPH_FILE_PATH)
        plt.show()

    def predict(self, valid):
        """
        Predict LSTM model on test data.

        Args:
          valid (pd.DataFrame): test dataframe

        Returns:
          LSTM_test_pred: predicted sentiment labels for test dataset
          LSTM_y_probs: probabilities of the predicted sentiment labels
        """

        self.model = load_model(self.saved_model_path)
        X_val_vect = self.get_word_vectors(valid)
        # Use the trained model to make predictions on the val data

        LSTM_y_pred = self.model.predict(X_val_vect)

        # keep probabilities for the positive outcome only
        LSTM_y_probs = LSTM_y_pred[:, 1]
        LSTM_test_pred = []

        for pred in LSTM_y_pred:
            if pred[0] >= 0.5:
                LSTM_test_pred.append(0)
            else:
                LSTM_test_pred.append(1)

        return LSTM_test_pred, LSTM_y_probs

    def evaluate(self, valid):
        """
        Evaluate LSTM model performance on valid data.

        Args:
          valid (pd.DataFrame): valid dataframe

        Returns:
          ap: average precision score
          pr_auc: precision recall area under curve score
        """
        LSTM_test_pred, LSTM_y_probs = self.predict(valid)
        y_test = tensorflow.keras.utils.to_categorical(valid[self.target_col])

        # get actual test response
        test_actual = []

        for pred in y_test:
            if pred[0] >= 0.5:
                test_actual.append(0)
            else:
                test_actual.append(1)

        LSTM_precision, LSTM_recall, LSTM_thresholds = precision_recall_curve(
            test_actual, LSTM_y_probs)

        ap = average_precision_score(test_actual,  LSTM_y_probs)
        pr_auc = auc(LSTM_recall, LSTM_precision)

        return ap, pr_auc
