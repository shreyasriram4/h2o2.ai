import re

import contractions
import nltk
import pandas as pd
from nltk.corpus import stopwords
import warnings

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def convert_sentiment_df(df: pd.DataFrame,
                         src_col: str = "Sentiment",
                         dst_col: str = "sentiment") -> pd.DataFrame:
    """
    Convert sentiments in a given dataframe from string format ('positive'
    and 'negative') to integer format (1 and 0)

    Args:
        df (pd.Dataframe): input dataframe with sentiment column
        src_col (str): column name of input sentiment column
        dst_col (str): expected column name of output sentiment column

    Returns:
        df (pd.Dataframe): dataframe consisting of sentiment column
        where entries are in binary integer format (1 and 0)
    """
    df[dst_col] = df[src_col].apply(
        lambda x: 1 if x == "positive" else 0
    )

    if src_col != dst_col:
        df = df.drop([src_col], axis=1)

    return df


def expand_contractions_df(df: pd.DataFrame,
                           src_col: str = "cleaned_text",
                           dst_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Expand contractions in a text column of a given dataframe using
    expand_contractions_text helper function

    Args:
        df (pd.Dataframe): input dataframe with text column
        src_col (str): column name of input text column with contractions
        dst_col (str): column name of output text column with
        contractions expande

    Returns:
        df (pd.Dataframe): dataframe consisting of text column
        with contractions explanded
    """
    df[dst_col] = df[src_col].apply(expand_contractions_text)
    return df


def expand_contractions_text(text: str) -> str:
    """
    Expand contractions in a string
    e.g.:
    can't -> cannot
    asap -> as soon as possible

    Args:
        text (str): text containing contractions

    Returns:
        output_text (str): text with contractions expanded
    """
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    output_text = ' '.join(expanded_words)
    return output_text


def lowercase_string_df(df: pd.DataFrame,
                        src_col: str = "cleaned_text",
                        dst_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Convert text in a text column of a given dataframe to lowercase

    Args:
        df (pd.Dataframe): input dataframe with text column
        src_col (str): column name of input text column
        dst_col (str): column name of output text column in lowercase
        format

    Returns:
        df (pd.Dataframe): dataframe consisting of text column
        converted to lowercase
    """
    df[dst_col] = df[src_col].apply(
        lambda text: text.lower()
    )
    return df


def remove_numbers_df(df: pd.DataFrame,
                      src_col: str = "cleaned_text",
                      dst_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Removes numbers in a text column of dataframe

    Args:
        df (pd.Dataframe): input dataframe with text column
        src_col (str): column name of input text column
        dst_col (str): column name of output text column with numbers
        removed

    Returns:
        df (pd.Dataframe): dataframe consisting of text column
        with numbers removed
    """
    df[dst_col] = df[src_col].apply(
        lambda text: ''.join([i for i in text if not i.isdigit()])
    )
    return df


def remove_punctuations_df(df: pd.DataFrame,
                           src_col: str = "cleaned_text",
                           dst_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Removes punctuation in a text column of dataframe

    Args:
        df (pd.Dataframe): input dataframe with text column
        src_col (str): column name of input text column
        dst_col (str): column name of output text column with punctuation
        removed

    Returns:
        df (pd.Dataframe): dataframe consisting of text column
        with punctuation removed
    """
    df[dst_col] = df[src_col].apply(
        lambda text: re.sub(r'[^\w\s]', ' ', text)
    )
    return df


def remove_stopwords_df(df: pd.DataFrame,
                        src_col: str = "cleaned_text",
                        dst_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Removes stopwords in a text column of dataframe

    Args:
        df (pd.Dataframe): input dataframe with text column
        src_col (str): column name of input text column
        dst_col (str): column name of output text column with stopwords
        removed

    Returns:
        df (pd.Dataframe): dataframe consisting of text column
        with stopwords removed
    """
    df[dst_col] = df[src_col].apply(
        remove_stopwords_text
    )
    return df


def remove_stopwords_text(text: str) -> str:
    """
    Removes stopwords in a string
    Stopwords are obtained from nltk.corpus
    e.g.: the, a, is, had, etc

    Args:
        text (str): text containing stopwords

    Returns:
        output_text (str): text with stopwords removed
    """
    output_text = " ".join([w for w in text.split(" ") if w not in STOP_WORDS])
    return output_text


def remove_trailing_leading_spaces_df(df: pd.DataFrame,
                                      src_col: str = "cleaned_text",
                                      dst_col: str = "cleaned_text") \
        -> pd.DataFrame:
    """
    Removes trailing leading spaces in a text column of dataframe.
    e.g.
    " The food was good" -> "The food was good"

    Args:
        df (pd.Dataframe): input dataframe with text column
        src_col (str): column name of input text column
        dst_col (str): column name of output text column with trailing
        leading spaces removed

    Returns:
        df (pd.Dataframe): dataframe consisting of text column
        with trailing leading spaces removed
    """
    df[dst_col] = df[src_col].apply(
        lambda text: text.strip()
    )
    return df


def rename_column_df(df: pd.DataFrame, src_col: str, dst_col: str) \
        -> pd.DataFrame:
    """
    Renames a text column in a given dataframe

    Args:
        df (pd.Dataframe): input dataframe
        src_col (str): input column name
        dst_col (str): output expected column name

    Returns:
        df (pd.Dataframe): dataframe consisting of a column
        with name changed
    """
    df = df.rename(columns={src_col: dst_col})
    return df


def replace_multiple_spaces_df(df: pd.DataFrame,
                               src_col: str = "cleaned_text",
                               dst_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Removes multiple spaces in a text column of dataframe.
    e.g.
    "The   food was good" -> "The food was good"

    Args:
        df (pd.Dataframe): input dataframe with text column
        src_col (str): column name of input text column
        dst_col (str): column name of output text column with multiple
        spaces removed

    Returns:
        df (pd.Dataframe): dataframe consisting of text column
        with multiple spaces removed
    """

    df[dst_col] = df[src_col].apply(
        lambda text: re.sub(' +', ' ', text)
    )
    return df


def strip_html_tags_df(df: pd.DataFrame,
                       src_col: str = "cleaned_text",
                       dst_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Removes HTML tags in a text column of dataframe.
    e.g.
    "<p>The food was good</p><br>" -> "The food was good"

    Args:
        df (pd.Dataframe): input dataframe with text column
        src_col (str): column name of input text column
        dst_col (str): column name of output text column with
        HTML tags removed

    Returns:
        df (pd.Dataframe): dataframe consisting of text column
        with HTML tags removed
    """
    df[dst_col] = df[src_col].apply(
        lambda text: re.sub('<[^<]+?>', ' ', text)
    )
    return df


def remove_empty_reviews_df(df: pd.DataFrame,
                            src_col: str = "cleaned_text") -> pd.DataFrame:
    """
    Filters empty rows out of the dataframe based on input text
    column of choice

    Args:
        df (pd.Dataframe): input dataframe with text column
        src_col (str): column name of text column that potentially
        contains empty strings ("")

    Returns:
        df (pd.Dataframe): filtered dataframe with empty rows removed
    """
    df = df[df[src_col] != ""]
    return df
