import contractions
import pandas as pd
import re
import string
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

def convert_sentiment_df(df: pd.DataFrame, src_col: str = "Sentiment", dst_col: str = "sentiment") -> pd.DataFrame:
    df[dst_col] = df[src_col].apply(
        lambda x: 1 if x == "positive" else 0
        )
    
    if src_col != dst_col:
        df.drop(src_col, axis = 1)

    return df

def expand_contractions_df(df: pd.DataFrame, src_col: str = "cleaned_text", dst_col: str = "cleaned_text") -> pd.DataFrame:    
    df[dst_col] = df[src_col].apply(expand_contractions_text)
    return df

def expand_contractions_text(text: str) -> str:
    expanded_words = []   
    for word in text.split():
        expanded_words.append(contractions.fix(word))  
        
    return ' '.join(expanded_words)

def lowercase_string_df(df: pd.DataFrame, src_col: str = "cleaned_text", dst_col: str = "cleaned_text") -> pd.DataFrame:
    df[dst_col] = df[src_col].apply(
        lambda text: text.lower()
        )
    return df

def remove_numbers_df(df: pd.DataFrame, src_col: str = "cleaned_text", dst_col: str = "cleaned_text") -> pd.DataFrame:
    df[dst_col] = df[src_col].apply(
        lambda text: ''.join([i for i in text if not i.isdigit()])
        )
    return df

def remove_punctuations_df(df: pd.DataFrame, src_col: str = "cleaned_text", dst_col: str = "cleaned_text") -> pd.DataFrame:
    df[dst_col] = df[src_col].apply(
        lambda text: text.translate(str.maketrans('', '', string.punctuation))
        )
    return df

def remove_stopwords_df(df: pd.DataFrame, src_col: str = "cleaned_text", dst_col: str = "cleaned_text") -> pd.DataFrame:
    df[dst_col] = df[src_col].apply(
        remove_stopwords_text
    )
    return df

def remove_stopwords_text(text: str) -> str:
    return " ".join([w for w in text.split(" ") if not w in STOP_WORDS])

def replace_multiple_spaces_df(df: pd.DataFrame, src_col: str = "cleaned_text", dst_col: str = "cleaned_text") -> pd.DataFrame:
    df[dst_col] = df[src_col].apply(
        lambda text: re.sub(' +', ' ', text)
    )
    return df

def strip_html_tags_df(df: pd.DataFrame, src_col: str = "cleaned_text", dst_col: str = "cleaned_text") -> pd.DataFrame:
    df[dst_col] = df[src_col].apply(
        lambda text: re.sub('<[^<]+?>', ' ', text)
        )
    return df
