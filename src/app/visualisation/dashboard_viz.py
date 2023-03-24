import os

import pandas as pd
import plotly.express as px
import numpy as np
import yaml
from plotly import graph_objects as go
from plotly import io as pio

from visualise_topics import visualise_top_words

# with open(os.path.join(os.getcwd(),"config.yml"), "r") as ymlfile:
#     CONFIG_PARAMS = yaml.safe_load(ymlfile)

def update_chart(fig):

    return fig.update(
        layout=go.Layout(margin=dict(t=40, r=0, b=40, l=0), legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.01))
    )

def reformat_data(data):
        data = data.assign(sentiment = data.sentiment.map({1: "positive", 0: "negative"}))
        data["date"] = pd.to_datetime(data["date"])
        return data

async def sentiment_pie_chart(data):
        fig = px.pie(data, names='sentiment', title='Overall Sentiments', color_discrete_sequence=px.colors.qualitative.Safe[0:2])
        html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
        return html

def sentiment_line_chart_over_time(data):
    freq_df = data.groupby(['date', 'sentiment'], as_index = False).size()
    fig = px.line(freq_df, x="date", y="size", color = "sentiment", title="Sentiments over Time", color_discrete_sequence=px.colors.qualitative.Safe[0:2][::-1]) 
    return fig

def topics_bar_chart(data):
    freq_df = data.groupby(['pred_topic_label', 'sentiment'], as_index = False).size()
    freq_df['pct'] = freq_df.groupby('pred_topic_label', group_keys = False)['size'].apply(lambda x: np.round(x*100/x.sum(), 1))
    fig = px.bar(freq_df, x='pred_topic_label', y='pct', color = 'sentiment', color_discrete_sequence=px.colors.qualitative.Safe[0:2][::-1])
    fig.update_layout(xaxis={"dtick":1})
    return fig

def topics_line_chart_over_time(data):
    freq_df = data.groupby(['date', 'pred_topic_label'], as_index = False).size()
    fig = px.area(freq_df, x="date", y="size", color = "pred_topic_label", title="Topics over Time", color_discrete_sequence=px.colors.qualitative.Safe) 
    return fig

def topics_pie_chart(data):
    fig = px.pie(data, 'pred_topic_label', title = "Frequency of Topics", color_discrete_sequence=px.colors.qualitative.Safe)
    return fig

def visualise_all_topics(data):
    return visualise_top_words(data, labels = CONFIG_PARAMS["labels"], specific = False, custom_sw = CONFIG_PARAMS["custom_stopwords"])

def visualise_all_topics(data, topic):
    return visualise_top_words(data, labels = topic, specific = True, custom_sw = CONFIG_PARAMS["custom_stopwords"])
    
def extract_top_reviews(data):
    return "in progress"




