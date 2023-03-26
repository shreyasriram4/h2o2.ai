import os

import pandas as pd
import plotly.express as px
import numpy as np
import yaml

from plotly import graph_objects as go
from plotly import io as pio

from src.visualisation.visualise_topics import visualise_top_words

with open(os.path.join(os.getcwd(),"config.yml"), "r") as ymlfile:
    CONFIG_PARAMS = yaml.safe_load(ymlfile)

def update_chart(fig):

    return fig.update(
        layout=go.Layout(margin=dict(t=40, r=0, b=40, l=0), legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01))
    )

def reformat_data(data):
    data = data.assign(sentiment = data.sentiment.map({1: "positive", 0: "negative"}))
    data["date"] = pd.to_datetime(data["date"])
    return data

async def sentiment_pie_chart(data):
    fig = px.pie(data, names='sentiment', title='Overall Sentiments', color_discrete_sequence=px.colors.qualitative.Plotly[1:3])
    # fig = px.pie(data, names='sentiment', title='Overall Sentiments', color_discrete_sequence=['green', 'red'])
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def sentiment_line_chart_over_time(data):
    freq_df = data.groupby(['date', 'sentiment'], as_index = False).size()
    fig = px.line(freq_df, x="date", y="size", color = "sentiment", title="Sentiments over Time", color_discrete_sequence=px.colors.qualitative.Safe[0:2][::-1]) 
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def topics_bar_chart(data):
    freq_df = data.groupby(['topic', 'sentiment'], as_index = False).size()
    freq_df['pct'] = freq_df.groupby('topic', group_keys = False)['size'].apply(lambda x: np.round(x*100/x.sum(), 1))
    fig = px.bar(freq_df, x='topic', y='pct', color = 'sentiment', color_discrete_sequence=px.colors.qualitative.Safe[0:2][::-1])
    update_chart(fig)
    fig.update_layout(xaxis={"dtick":1})
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def topics_line_chart_over_time(data):
    freq_df = data.groupby(['date', 'topic'], as_index = False).size()
    fig = px.area(freq_df, x="date", y="size", color = "topic", title="Topics over Time", color_discrete_sequence=px.colors.qualitative.Safe) 
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def topics_pie_chart(data):
    fig = px.pie(data, 'topic', title = "Frequency of Topics", color_discrete_sequence=px.colors.qualitative.Safe)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def visualise_all_topics(data):
    return visualise_top_words(data, labels = CONFIG_PARAMS["labels"], specific = False, custom_sw = CONFIG_PARAMS["custom_stopwords"])

async def visualise_all_topics(data, topic):
    return visualise_top_words(data, labels = topic, specific = True, custom_sw = CONFIG_PARAMS["custom_stopwords"])
    
async def extract_top_reviews(data, topic):
    topic_df = data[data["pred_topic_label"] == topic]
    topic_sliced = list(topic_df.sort_values(by = "score").head(5)["partially_cleaned_text"])
    return topic_sliced

async def topics_bar_chart_by_month(data):
    data['year_month'] = data['date'].dt.to_period('M').astype('string')
    print(data)
    freq_df = data.groupby(['year_month', 'topic'], as_index = False).size()
    fig = px.area(freq_df, x="year_month", y="size", color = "topic", title="Topics over Time", color_discrete_sequence=px.colors.qualitative.Safe) 
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def extract_num_topics(data):
    count_df = data.groupby(['topic'], as_index = False).size()
    return "in progress"




# def reformat_data(data):
#     data = data.assign(sentiment = data.sentiment.map({1: "positive", 0: "negative"}))
#     data["date"] = pd.to_datetime(data["date"])
#     return data

# async def sentiment_pie_chart(data):
#     fig = px.pie(data, names='sentiment', title='Overall Sentiments', color_discrete_sequence=px.colors.qualitative.Safe[0:2])
#     update_chart(fig)
#     html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
#     return html

# def sentiment_line_chart_over_time(data):
#     freq_df = data.groupby(['date', 'sentiment'], as_index = False).size()
#     fig = px.line(freq_df, x="date", y="size", color = "sentiment", title="Sentiments over Time", color_discrete_sequence=px.colors.qualitative.Safe[0:2][::-1]) 
#     return fig

# def topics_bar_chart(data):
#     freq_df = data.groupby(['pred_topic_label', 'sentiment'], as_index = False).size()
#     freq_df['pct'] = freq_df.groupby('pred_topic_label', group_keys = False)['size'].apply(lambda x: np.round(x*100/x.sum(), 1))
#     fig = px.bar(freq_df, x='pred_topic_label', y='pct', color = 'sentiment', color_discrete_sequence=px.colors.qualitative.Safe[0:2][::-1])
#     fig.update_layout(xaxis={"dtick":1})

#     return fig

# def topics_line_chart_over_time(data):
#     freq_df = data.groupby(['date', 'pred_topic_label'], as_index = False).size()
#     fig = px.area(freq_df, x="date", y="size", color = "pred_topic_label", title="Topics over Time", color_discrete_sequence=px.colors.qualitative.Safe) 
#     return fig

# def topics_pie_chart(data):
#     fig = px.pie(data, 'pred_topic_label', title = "Frequency of Topics", color_discrete_sequence=px.colors.qualitative.Safe)
#     html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
#     return fig

# def visualise_all_topics(data):
#     return visualise_top_words(data, labels = CONFIG_PARAMS["labels"], specific = False, custom_sw = CONFIG_PARAMS["custom_stopwords"])

# def visualise_all_topics(data, topic):
#     return visualise_top_words(data, labels = topic, specific = True, custom_sw = CONFIG_PARAMS["custom_stopwords"])
    
# def extract_top_reviews(data):
#     return "in progress"