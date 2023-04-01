import os

import pandas as pd
import plotly.express as px
import numpy as np
import yaml
import collections

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


async def sentiment_pie_chart(data):
    '''
    Plots a pie chart to show distribution of positive and negative
    sentiments in the data.
    ------------
    Parameters:
        data (dataframe): dataframe
    Returns:
        html (html): html of the plotly figure
    '''
    fig = px.pie(data, names='sentiment',
                 color_discrete_sequence=px.colors.qualitative.Plotly[1:3],
                 category_orders={'sentiment': ['negative', 'positive']})
    # fig = px.pie(data, names='sentiment',
    # color_discrete_sequence=px.colors.qualitative.Plotly[1:3])
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True,
                       include_plotlyjs="cdn")
    return html


async def sentiment_line_chart_over_time(data):
    '''
    Plots a line plot of positive and negative sentiments in the data overtime.
    ------------
    Parameters:
        data (dataframe): dataframe
    Returns:
        html (html): html of the plotly figure
    '''
    freq_df = data.groupby(['date', 'sentiment'], as_index=False).size()
    fig = px.line(freq_df, x="date", y="size", color="sentiment",
                  labels={'date': 'Date', 'size': 'Number of Reviews'},
                  color_discrete_sequence=px.colors.qualitative.Plotly[1:3],
                  category_orders={'sentiment': ['negative', 'positive']})
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True,
                       include_plotlyjs="cdn")
    return html


async def topics_bar_chart(data):
    '''
    Bar chart to visualise of the distribution of positive and
    negative sentiments for each topic in the data.
    ------------
    Parameters:
        data (dataframe): dataframe
    Returns:
        html (html): html of the plotly figure
    '''
    freq_df = data.groupby(['topic', 'sentiment'], as_index=False).size()
    freq_df['pct'] = freq_df.groupby('topic',
                                     group_keys=False)['size'].apply(
                                    lambda x: np.round(x*100/x.sum(), 1))
    fig = px.bar(freq_df, x='topic', y='pct', color='sentiment',
                 labels={"topic": "Topic", 'pct': "Percentage(%)"},
                 color_discrete_sequence=px.colors.qualitative.Plotly[1:3],
                 category_orders={'sentiment': ['negative', 'positive']})
    update_chart(fig)
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


async def topics_pie_chart(data):
    '''
    Pie chart showing distribution of topics in the data.
    ------------
    Parameters:
        data (dataframe): dataframe
    Returns:
        html (html): html of the plotly figure
    '''
    fig = px.pie(data, 'topic',
                 category_orders={'topic': CONFIG_PARAMS["labels"]})
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True,
                       include_plotlyjs="cdn")
    return html


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


async def visualise_all_topics_playground(data, topic):
    '''
    Barcharts showing top key words in the selected topic.
    ------------
    Parameters:
        data (dataframe): dataframe
        topic (str): string of topic
    Returns:
        html (html): html of the plotly figure
    '''
    fig = visualise_top_words(data, topics=topic,
                              specific=True,
                              custom_sw=CONFIG_PARAMS["custom_stopwords"])
    update_chart(fig)
    fig.update_yaxes(dtick=1)
    html = pio.to_html(fig, config=None, auto_play=True,
                       include_plotlyjs="cdn")
    return html


# async def extract_top_reviews(data, topic, sentiment):
#     '''
#     Extract the list of reviews in the selected topic.
#     ------------
#     Parameters:
#         data (dataframe): dataframe
#         topic (str): string of topic
#         sentiment (str): sentiment (e.g 'positive' or 'negative')
#     Returns:
#         html (html): html of the plotly figure
#     '''
#     topic_df = data[(data["topic"] == topic) &
#                     (data["sentiment"] == sentiment)]
#     topic_sliced = list(
#         topic_df.sort_values(by="sentiment_prob")["partially_cleaned_text"])
#     return topic_sliced


async def extract_top_topic_reviews(data, topic):
    '''
    Barcharts showing top key words in the selected topic.
    ------------
    Parameters:
        data (dataframe): dataframe
        topic (str): string of topic
    Returns:
        html (html): html of the plotly figure
    '''
    topic_df = data[(data["topic"] == topic)]
    topic_sliced = topic_df.sort_values(by="sentiment_prob")[[
        "partially_cleaned_text", "sentiment"]]
    return topic_sliced


# async def topics_line_chart_by_quarter(data):
#     '''
#     Line chart plotting the distribution of topics overtime.
#     ------------
#     Parameters:
#         data (dataframe): dataframe
#     Returns:
#         html (html): html of the plotly figure
#     '''
#     data['year_quarter'] = data['date'].dt.to_period('Q').astype('string')
#     freq_df = data.groupby(['year_quarter', 'topic'], as_index=False).size()
#     fig = px.area(freq_df, x="year_quarter", y="size", color="topic",
#                   category_orders={'topic': CONFIG_PARAMS["labels"]},
#                   labels={"topic": "Topic", 'size': "Number of reviews"})
#     update_chart(fig)
#     html = pio.to_html(fig, config=None, auto_play=True,
#                        include_plotlyjs="cdn")
#     return html


async def topics_bar_chart_over_time(data, time_frame=None):
    '''
    Bar chart plotting the distribution of topics against calendar quarters.
    ------------
    Parameters:
        data (dataframe): dataframe
        time_frame (str): String of the timeframe required 
                        (eg. "M" for months or "Q" for quarter)
    Returns:
        html (html): html of the plotly figure
    '''
    if time_frame is not None:
        data['date_frame'] = data['date'].dt.to_period(
                                            time_frame).astype('string')
        freq_df = data.groupby(['date_frame', 'topic'], as_index=False).size()
        fig = px.bar(freq_df, x='date_frame', y='size', color='topic',
                     category_orders={'topic': CONFIG_PARAMS["labels"]},
                     labels={"topic": "Topic", 'size': "Number of reviews",
                                'date_frame': 'Time'},
                     barmode='group')
    else:
        freq_df = data.groupby(['date', 'topic'], as_index=False).size()
        fig = px.bar(freq_df, x='date', y='size', color='topic',
                     category_orders={'topic': CONFIG_PARAMS["labels"]},
                     labels={"topic": "Topic", 'size': "Number of reviews"},
                     barmode='group')
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True,
                       include_plotlyjs="cdn")
    return html


async def extract_num_topics(data):
    '''
    Extracts the number of topics.
    ------------
    Parameters:
        data (dataframe): dataframe
    Returns:
        num_topics (str): number of topics
    '''
    num_topics = data['topic'].nunique()
    return str(num_topics)


# async def extract_most_freq_words(data, positive=True):
#     '''
#     Extracts the most frequent words among positive and negative
#     sentiments respectively.
#     ------------
#     Parameters:
#         data (dataframe): dataframe
#     Returns:
#         word (list): list of most frequently appearing words
#     '''
#     if positive is True:
#         df = data[data['sentiment'] == 'positive']
#     else:
#         df = data[data['sentiment'] == 'negative']
#     my_stop_words = list(text.ENGLISH_STOP_WORDS.union(
#         CONFIG_PARAMS["custom_stopwords"]))
#     cv = CountVectorizer(stop_words=my_stop_words)
#     bow = cv.fit_transform(df['cleaned_text'])
#     word_freq = dict(zip(cv.get_feature_names(),
#                          np.asarray(bow.sum(axis=0)).ravel()))
#     word_counter = collections.Counter(word_freq)
#     word_counter_df = pd.DataFrame(word_counter.most_common(20),
#                                    columns=['word', 'freq'])
#     return word_counter_df['word']


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
