import os
import pandas as pd
import numpy as np
import yaml
import collections

import plotly.express as px
from plotly import graph_objects as go
from plotly import io as pio

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

from src.visualisation.visualise_topics import visualise_top_words

with open(os.path.join(os.getcwd(), "config.yml"), "r") as ymlfile:
    CONFIG_PARAMS = yaml.safe_load(ymlfile)


def update_chart(fig):
    '''
    Updates the figure with the required layout.
    ------------
    Parameters:
        fig (graph object): plotly figure
    Returns:
        fig (graph object): plotly figure
    '''
    return fig.update(
        layout=go.Layout(margin=dict(t=20, r=5, b=20, l=5),
                         legend=dict(yanchor="bottom", y=1, xanchor="right",
                                     x=1.2),
                         title_font_family='Verdana',
                         font_family='Century Gothic',
                         template="plotly_white"))


def reformat_data(data):
    '''
    Mapping the sentiment column with values 1 to positive and 0 to negative.
    Format the date column to datetime.
    ------------
    Parameters:
        data (dataframe): dataframe consisting sentiment and date columns
    Returns:
        data (dataframe): dataframe
    '''
    data = data.assign(sentiment=data.sentiment.map({1: "positive",
                                                     0: "negative"}))
    data["date"] = pd.to_datetime(data["date"])
    return data


def sentiment_pie_chart(data):
    fig = px.pie(data, names='sentiment', title='Overall Sentiments',
                 color_discrete_sequence=px.colors.qualitative.Safe[0:2])
    return fig


def sentiment_line_chart_over_time(data):
    freq_df = data.groupby(['date', 'sentiment'],
                           as_index=False).size()
    fig = px.line(
        freq_df, x="date", y="size", color="sentiment",
        title="Sentiments over Time",
        color_discrete_sequence=px.colors.qualitative.Safe[0:2][::-1])
    return fig


def topics_bar_chart(data):
    freq_df = data.groupby(['pred_topic_label', 'sentiment'],
                           as_index=False).size()
    freq_df['pct'] = freq_df.groupby(
        'pred_topic_label', group_keys=False)['size'].apply(
            lambda x: np.round(x*100/x.sum(), 1)
            )
    fig = px.bar(freq_df, x='pred_topic_label', y='pct', color='sentiment',
                 color_discrete_sequence=px.colors.qualitative.Safe[0:2][::-1])
    fig.update_layout(xaxis={"dtick": 1})
    html = pio.to_html(fig, config=None, auto_play=True,
                       include_plotlyjs="cdn")
    return html


async def topics_line_chart_over_time(data):
    '''
    Plots distribution of topics overtime.
    ------------
    Parameters:
        data (dataframe): dataframe
    Returns:
        html (html): html of the plotly figure
    '''
    freq_df = data.groupby(['date', 'topic'], as_index=False).size()
    fig = px.area(freq_df, x="date", y="size", color="topic")
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True,
                       include_plotlyjs="cdn")
    return html


def topics_pie_chart(data):
    fig = px.pie(data, 'pred_topic_label', title="Frequency of Topics",
                 color_discrete_sequence=px.colors.qualitative.Safe)
    return fig


async def visualise_all_topics(data):
    '''
    Barcharts showing top key words in each topics.
    ------------
    Parameters:
        data (dataframe): dataframe
    Returns:
        html (html): html of the plotly figure
    '''
    fig = visualise_top_words(data, topics=CONFIG_PARAMS["labels"],
                              specific=False,
                              custom_sw=CONFIG_PARAMS["custom_stopwords"])
    update_chart(fig)
    fig.update_yaxes(dtick=1)
    html = pio.to_html(fig, config=None, auto_play=True,
                       include_plotlyjs="cdn")
    return html


def visualise_all_topics(data, topic):
    return visualise_top_words(data, labels=topic, specific=True,
                               custom_sw=CONFIG_PARAMS["custom_stopwords"])


async def get_subtopics(data, topic):
    '''
    Extracts the number of topics.
    ------------
    Parameters:
        data (dataframe): dataframe
        topic (str): string of topic
    Returns:
        html (html): html of the plotly figure
    '''
    print(data.columns)
    df = data[data['topic'] == topic]
    df = df.groupby(['subtopic'], as_index=False).size().sort_values('size')
    print(df)
    fig = px.bar(df, x="size", y="subtopic", orientation='h',
                 labels={'size': 'Number of Reviews', 'subtopic': 'Subtopic'})
    update_chart(fig)
    fig.update_yaxes(dtick=1)
    html = pio.to_html(fig, config=None, auto_play=True,
                       include_plotlyjs="cdn")
    return html
