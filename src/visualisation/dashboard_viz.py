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

with open(os.path.join(os.getcwd(),"config.yml"), "r") as ymlfile:
    CONFIG_PARAMS = yaml.safe_load(ymlfile)

def update_chart(fig):

    return fig.update(
        layout=go.Layout(margin=dict(t=20, r=5, b=20, l=5), 
                         legend=dict(yanchor="bottom", y=1, xanchor="right", x=1.2), 
                         title_font_family='Verdana',
                         font_family='Century Gothic',
                         template="plotly_white")
    )

def reformat_data(data):
    data = data.assign(sentiment = data.sentiment.map({1: "positive", 0: "negative"}))
    data["date"] = pd.to_datetime(data["date"])
    return data

async def sentiment_pie_chart(data):
    fig = px.pie(data, names='sentiment', title='Overall Sentiments', color_discrete_sequence=px.colors.qualitative.Plotly[1:3], category_orders={'sentiment':['negative', 'positive']})
    # fig = px.pie(data, names='sentiment', color_discrete_sequence=px.colors.qualitative.Plotly[1:3])
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def sentiment_line_chart_over_time(data):
    freq_df = data.groupby(['date', 'sentiment'], as_index = False).size()
    fig = px.line(freq_df, x="date", y="size", color = "sentiment", title="Sentiments over Time", color_discrete_sequence=px.colors.qualitative.Plotly[1:3]) 
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def topics_bar_chart(data):
    freq_df = data.groupby(['topic', 'sentiment'], as_index = False).size()
    freq_df['pct'] = freq_df.groupby('topic', group_keys = False)['size'].apply(lambda x: np.round(x*100/x.sum(), 1))
    fig = px.bar(freq_df, x='topic', y='pct', color = 'sentiment', title="Topics by Sentiment", labels={"topic": "Topic", 'pct':"Percentage(%)"}, color_discrete_sequence=px.colors.qualitative.Plotly[1:3])
    update_chart(fig)
    fig.update_layout(xaxis={"dtick":1})
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def topics_line_chart_over_time(data):
    freq_df = data.groupby(['date', 'topic'], as_index = False).size()
    fig = px.area(freq_df, x="date", y="size", color = "topic", title="Topics over Time") 
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def topics_pie_chart(data):
    # fig = px.pie(data, 'topic', title = "Frequency of Topics", color_discrete_sequence=px.colors.qualitative.Safe)
    fig = px.pie(data, 'topic', title = "Frequency of Topics", category_orders={'topic':CONFIG_PARAMS["labels_app"]})
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def visualise_all_topics(data):
    # fig = visualise_top_words(data, labels = CONFIG_PARAMS["labels"], specific = False, custom_sw = CONFIG_PARAMS["custom_stopwords"])
    fig = visualise_top_words(data, topics = CONFIG_PARAMS["labels_app"], specific = False, custom_sw = CONFIG_PARAMS["custom_stopwords"])
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def visualise_all_topics_playground(data, topic):
    fig = visualise_top_words(data, labels = topic, specific = True, custom_sw = CONFIG_PARAMS["custom_stopwords"])
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html
    
async def extract_top_reviews(data, topic):
    topic_df = data[data["topic"] == topic]
    topic_sliced = list(topic_df.sort_values(by = "score").head(5)["partially_cleaned_text"])
    return topic_sliced

async def topics_line_chart_by_quarter(data):
    data['year_quarter'] = data['date'].dt.to_period('Q').astype('string')
    freq_df = data.groupby(['year_quarter', 'topic'], as_index = False).size()
    # fig = px.bar(freq_df, x='year_month', y='size', color = 'topic', labels={"topic": "Topic", 'pct':"Percentage(%)"}, barmode='group')
    fig = px.area(freq_df, x="year_quarter", y="size", color = "topic", title="Topics over Time", category_orders={'topic':CONFIG_PARAMS["labels_app"]}, labels={"topic": "Topic", 'size':"Number of reviews"}) 
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def topics_bar_chart_over_time(data, time_frame=None):
    if time_frame != None:
        data['date_frame'] = data['date'].dt.to_period(time_frame).astype('string')
        freq_df = data.groupby(['date_frame', 'topic'], as_index = False).size()
        fig = px.bar(freq_df, x='date_frame', y='size', color = 'topic', title="Topics over Time", labels={"topic": "Topic", 'size':"Number of reviews"}, barmode='group')
    else:
        freq_df = data.groupby(['date', 'topic'], as_index = False).size()
        fig = px.bar(freq_df, x='date', y='size', color = 'topic', title="Topics over Time", labels={"topic": "Topic", 'size':"Number of reviews"}, barmode='group')
    update_chart(fig)
    html = pio.to_html(fig, config=None, auto_play=True, include_plotlyjs="cdn")
    return html

async def extract_num_topics(data):
    num_topics = data['topic'].nunique()
    return str(num_topics)

async def extract_most_freq_words_by_sentiment(data, positive=True):
    if positive==True:
        df = data[data['sentiment']=='positive']
    else:
        df = data[data['sentiment']=='negative']
    my_stop_words = list(text.ENGLISH_STOP_WORDS.union(CONFIG_PARAMS["custom_stopwords"]))
    cv = CountVectorizer(stop_words=my_stop_words)
    bow = cv.fit_transform(df['cleaned_text'])
    word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
    word_counter = collections.Counter(word_freq)
    word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
    return word_counter_df['word']