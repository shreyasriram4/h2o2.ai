import os
import pandas as pd
import numpy as np
import yaml
import collections

import plotly.express as px
from plotly import graph_objects as go
from plotly import io as pio

from src.visualisation.visualise_topics import visualise_top_words

with open(os.path.join(os.getcwd(), "config.yml"), "r") as ymlfile:
    CONFIG_PARAMS = yaml.safe_load(ymlfile)


def update_chart(fig):
    """
    Updates figure with the required layout.

    Args:
        fig (graph object): plotly figure

    Returns:
        fig (graph object): plotly figure with updated layout
    """
    return fig.update(
        layout=go.Layout(margin=dict(t=20, r=5, b=20, l=5),
                         legend=dict(yanchor="bottom", y=1, xanchor="right",
                                     x=1.2),
                         title_font_family='Verdana',
                         font_family='Century Gothic',
                         template="plotly_white"))


def reformat_data(data):
    """
    Mapping the sentiment column with values 1 to positive and 0 to negative.
    Format the date column to datetime.

    Args:
        data (pd.Dataframe): dataframe

    Returns:
        data (pd.Dataframe): dataframe with reformatted sentiment and
            date columns
    """
    data = data.assign(sentiment=data.sentiment.map({1: "positive",
                                                     0: "negative"}))
    data["date"] = pd.to_datetime(data["date"])
    return data


def sentiment_pie_chart(data):
    """
    Plots a pie chart to show distribution of positive and negative
    sentiments in the data.

    Args:
        data (pd.Dataframe): dataframe

    Returns:
        fig (graph object): plotly pie chart figure
    """
    fig = px.pie(data, names='sentiment',
                 color='sentiment',
                 color_discrete_map={
                    'negative': px.colors.qualitative.Plotly[1],
                    'positive': px.colors.qualitative.Plotly[2]})
    update_chart(fig)
    return fig


def sentiment_line_chart_over_time(data):
    """
    Plots a line chart of positive and negative sentiments over time.

    Args:
        data (pd.Dataframe): dataframe

    Returns:
        fig (graph object): plotly line chart figure
    """
    freq_df = data.groupby(['date', 'sentiment'], as_index=False).size()
    fig = px.line(freq_df, x="date", y="size",
                  labels={'date': 'Date', 'size': 'Number of Reviews'},
                  color="sentiment",
                  color_discrete_map={
                    'negative': px.colors.qualitative.Plotly[1],
                    'positive': px.colors.qualitative.Plotly[2]})
    update_chart(fig)
    return fig


def topics_bar_chart(data):
    """
    Bar chart to visualise the distribution of positive and
    negative sentiments for each topic in the data.

    Args:
        data (pd.Dataframe): dataframe

    Returns:
        fig (graph object): plotly bar chart figure
    """
    freq_df = data.groupby(['topic', 'sentiment'], as_index=False).size()
    freq_df['pct'] = freq_df.groupby('topic',
                                     group_keys=False)['size'].apply(
                                    lambda x: np.round(x*100/x.sum(), 1))
    fig = px.bar(freq_df, x='topic', y='pct',
                 labels={"topic": "Topic", 'pct': "Percentage(%)"},
                 color='sentiment',
                 color_discrete_map={
                    'negative': px.colors.qualitative.Plotly[1],
                    'positive': px.colors.qualitative.Plotly[2]})
    update_chart(fig)
    fig.update_layout(xaxis={"dtick": 1})
    return fig


def topics_line_chart_over_time(data):
    """
    Plots trend (number of reviews) of each topic over time.

    Args:
        data (pd.Dataframe): dataframe

    Returns:
        fig (graph object): plotly line chart figure
    """
    freq_df = data.groupby(['date', 'topic'], as_index=False).size()
    fig = px.area(freq_df, x="date", y="size", color="topic")
    update_chart(fig)
    return fig


def topics_pie_chart(data):
    """
    Pie chart showing distribution of topics in the data.

    Args:
        data (pd.Dataframe): dataframe

    Returns:
        fig (graph object): plotly pie chart figure
    """
    fig = px.pie(data, 'topic',
                 category_orders={'topic': CONFIG_PARAMS["labels"]})
    update_chart(fig)
    fig.update_layout(margin=dict(t=35, r=5, b=20, l=5),
                      legend=dict(
                        yanchor="top",
                        y=1.0,
                        xanchor="right",
                        x=1.4
                ))
    return fig


def visualise_all_topics(data):
    """
    Barcharts showing top key words in each topic.

    Args:
        data (pd.Dataframe): dataframe

    Returns:
        fig (graph object): plotly figure
    """
    fig = visualise_top_words(data, topics=list(data.topic.unique()),
                              specific=False,
                              custom_sw=CONFIG_PARAMS["custom_stopwords"])
    update_chart(fig)
    fig.update_yaxes(dtick=1)
    return fig


def visualise_all_topics_playground(data, topic):
    """
    Barcharts showing top key words in the selected topic.

    Args:
        data (pd.Dataframe): dataframe
        topic (str): string of topic

    Returns:
        fig (graph object): plotly figure
    """
    fig = visualise_top_words(data, topics=topic,
                              specific=True,
                              custom_sw=CONFIG_PARAMS["custom_stopwords"])
    update_chart(fig)
    fig.update_yaxes(dtick=1)
    return fig


async def extract_top_reviews(data, topic, sentiment):
    """
    Extract reviews in the selected topic based on predicted
    sentiment probability.

    Args:
        data (pd.Dataframe): dataframe
        topic (str): string of topic
        sentiment (str): sentiment (e.g 'positive' or 'negative')

    Returns:
        topic_sliced (list): list of reviews sorted by sentiment probability
            in descending order
    """
    topic_df = data[(data["topic"] == topic) &
                    (data["sentiment"] == sentiment)]
    topic_sliced = list(
        topic_df.sort_values(
            by="sentiment_prob", ascending=False)["partially_cleaned_text"])
    return topic_sliced


async def extract_top_topic_reviews(data, topic):
    """
    Extracts top reviews based on predicted sentiment probability
    in the selected topic.

    Args:
        data (pd.Dataframe): dataframe
        topic (str): string of topic

    Returns:
        topic_sliced (pd.Dataframe): dataframe consisting of review
            and predicted sentiment
    """
    topic_df = data[(data["topic"] == topic)]
    topic_sliced = topic_df.sort_values(
                        by="sentiment_prob", ascending=False
                        )[["partially_cleaned_text", "sentiment"]]
    return topic_sliced


def topics_bar_chart_over_time(data, time_frame=None):
    """
    Bar chart plotting the distribution of topics against specified
    time frame.

    Args:
        data (pd.Dataframe): dataframe
        time_frame (str): String of the timeframe required
            (eg. "M" for months or "Q" for quarter)

    Returns:
        fig (graph object): plotly bar chart figure
    """
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
    fig.update_layout(xaxis=dict(categoryorder='category ascending'))
    return fig


async def extract_num_topics(data):
    """
    Extracts the total number of topics.

    Args:
        data (pd.Dataframe): dataframe

    Returns:
        num_topics (str): number of topics
    """
    num_topics = data['topic'].nunique()
    return str(num_topics)


def get_subtopics(data, topic):
    """
    Distribution of subtopics under selected topic.

    Args:
        data (pd.Dataframe): dataframe
        topic (str): string representing topic

    Returns:
        fig (graph object): plotly figure
    """
    df = data[data['topic'] == topic]
    df = df.groupby(['subtopic'], as_index=False).size().sort_values('size')

    fig = px.bar(df, x="size", y="subtopic", orientation='h',
                 labels={'size': 'Number of Reviews', 'subtopic': 'Subtopic'})
    update_chart(fig)
    fig.update_yaxes(dtick=1)
    return fig


async def html_output(fig):
    """
    Converts plotly figure to html

    Args:
        fig (graph object): plotly figure

    Returns:
        html (html): html of plotly figure
    """

    html = pio.to_html(fig, config=None, auto_play=True,
                       include_plotlyjs="cdn")
    return html
