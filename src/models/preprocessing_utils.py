import string

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

class Preprocess:

    def __init__(self, model = 'en_core_web_sm', stop = STOP_WORDS):
        self.model = spacy.load(model)
        self.stop = stop

    def remove_punctuation_text(self, text):
        """
        
        Removes all punctuation in a given string using String library
        
        Args: 
            1. String containing unwanted punctuation (str)

        Returns:
            1. String without unwanted punctuation (str)

        """
        punctuationfree="".join([i for i in text if i not in string.punctuation])
        return punctuationfree


    def remove_punctuation_df(self, data, col):
        """
        
        Calls remove_punctuation_text function on each row in a particular column in dataframe, to add a new column that 
        removes all punctuation in a given column using Pandas library

        Args: 
            1. Dataframe (pandas.core.frame.DataFrame)
            2. Column in Dataframe with punctuation to be removed (str)

        Returns:
            1. Updated column with all punctuation removed (pandas.core.series.Series)

        """
        return data[col].apply(self.remove_punctuation_text)


    def lowercase_df(self, data, col):
        """

        Calls lower() function in String class on each row in a particular column in dataframe, to convert all
        text to lowercase in a given column
        
        Args: 
            1. Dataframe (pandas.core.frame.DataFrame)
            2. Column in Dataframe to be converted to lowercase (str)

        Returns:
            1. Updated column with all text converted to lowercase (pandas.core.series.Series)

        """
        return data[col].apply(lambda x: x.lower())


    def lemmatization_text(self, text):
        """

        Lemmatizes text based on Spacy's English module

        Args:
            1. Raw text (str)

        Returns:
            1. Lemmatized text (str) 

        """
        doc = self.model(text)
        tokens = [token.lemma_ for token in doc]

        return tokens

    def lemmatization_df(self, data, col):
        """

        Applies lemmatization_text to a all entries in a particular column 'col' in dataframe, to lemmatize text

        Args:
            1. Dataframe (pandas.core.frame.DataFrame)
            2. Column in dataframe to be lemmatized (str)
        
        Returns:
            1. Updated column with all text lemmatized (pandas.core.series.Series)
        
        """
        return data[col].apply(self.lemmatization_text)

    def tokenize_text(self, text):
        """
        
    Tokenizes text based on Spacy's English module

        Args:
            1. Text to tokenize (str)

        Returns:
            1. Tokenized text (str) 

        """
        doc = self.model(text)
        tokens = [token.text for token in doc]
        return tokens

    def tokenization_df(self, data, col):
        """

        Applies tokenization_text to a all entries in a particular column 'col' in dataframe

        Args:
            1. Dataframe (pandas.core.frame.DataFrame)
            2. Column in dataframe to be tokenized (str)
        
        Returns:
            1. Updated column with all text tokenized (pandas.core.series.Series)
        
        """
        return data[col].apply(self.tokenize_text)

    def remove_stopwords_text(self, text):
        """
        
        Removes stopwords in a string based on if a given word in string appears in "stop" dictionary
        
        Args:
            1. Text containing stopwords (str)

        Returns:
            1. Text with stopwords removed (str)
        
        """
        tokenized_text = self.tokenize_text(text)
        filtered_text = " ".join([token for token in tokenized_text if token not in self.stop])
        return filtered_text

    def remove_stopwords_df(self, data, col):
        """

        Applies remove_stopwords to all entries in particular column 'col' in dataframe, to remove stopwords

        Args:
            1. Dataframe (pandas.core.frame.DataFrame)
            2. Column in dataframe to have stopwords removed (str)

        Returns:
            1. Updated column with each column having stopwords removed (pandas.core.series.Series)

        """
        return data[col].apply(self.remove_stopwords_text)

    def remove_numbers_text(self, text):
        """
        
        Removes all numbers in a given string using String library
        
        Args: 
            1. String containing unwanted numbers (str)
        Returns:
            1. String without unwanted numbers (str)
        """
        numbersfree="".join([i for i in text if i not in string.digits])
        return numbersfree


    def remove_numbers_df(self, data, col):
        """
        
        Calls remove_numbers_text function on each row in a particular column in dataframe, to add a new column that 
        removes all numbers in a given column using Pandas library
        Args: 
            1. Dataframe (pandas.core.frame.DataFrame)
            2. Column in Dataframe with numbers to be removed (str)
        Returns:
            1. Updated column with all numbers removed (pandas.core.series.Series)
        """
        return data[col].apply(self.remove_numbers_text)