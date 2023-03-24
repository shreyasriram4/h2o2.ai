from h2o_wave import main, app, Q, ui, on, handle_on, data
from helper import add_card, clear_cards
import pandas as pd
from plotly import graph_objects as go
from plotly import io as pio
# from charts import reformat_data, sentiment_pie_chart
from visualisation.dashboard_viz import reformat_data, sentiment_pie_chart, sentiment_line_chart_over_time

# def load_data():
#     data = pd.read_csv('../../data/processed/reviews.csv')
#     data.date = pd.to_datetime(data.date)
#     data['year_month'] = data.date.dt.to_period('M')
#     return data

df = pd.read_csv('../../data/processed/reviews.csv')
df = reformat_data(df)

@on('#sentiments')
async def page2(q: Q):
    q.page['sidebar'].value = '#sentiments'
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    # fig = await sentiment_pie_chart(data=df)
    add_card(q, 'piechart1', ui.frame_card(
        box='horizontal_sentiment',
        title="Overall Sentiment",
        content=await sentiment_pie_chart(data=df),
        ))
    
    add_card(q, 'piechart2', ui.frame_card(
        box='horizontal_sentiment',
        title="Sentiment Overtime",
        content=await sentiment_line_chart_over_time(data=df),
    ))
    
    # add_card(q, 'chart3', ui.markdown_card(
    #     box='horizontal',
    #     title='Sentiment by Topic',
    #     content='Placeholder for now',
    # ))
    
    # print(html)



    await q.page.save()

    # def sentiment_over_time_df(df):
#     df['date'] = df['date'].dt.strftime("%Y-%m-%d")
#     time_df = df.groupby(['date', 'sentiment']).count().reset_index()
#     return time_df

# time_df = sentiment_over_time_df(df)


    # pos = (df['sentiment']==1).sum()
    # neg = (df['sentiment']==0).sum()
    # total = pos+neg
    # add_card(q, 'piechart', ui.wide_pie_stat_card(
    #     box='horizontal',
    #     title='Overall Sentiment',
    #     pies=[
    #         ui.pie(label='Positive', value=f'{round(pos/total * 100, 1)}%', fraction=pos/total, color='$green'),
    #         ui.pie(label='Negative', value=f'{round(neg/total * 100, 1)}%', fraction=neg/total, color='$red'),
    #     ]
    # ))

    # add_card(q, 'time_trend', ui.plot_card(
    #     box='horizontal',
    #     title='Sentiment Over Time',
    #     data=data(fields=time_df.columns.tolist(),rows = time_df.values.tolist()),
    #     plot = ui.plot(marks=[ui.mark(type='line',x='=date',y='=partially_cleaned_text', color='=sentiment', color_range='$green $red')])
    # ))

    # add_card(q, 'chart3', ui.markdown_card(
    #     box='horizontal1',
    #     title='Sentiment by Topic',
    #     content='Placeholder for now',
    # ))

    # add_card(q, 'chart4', ui.markdown_card(
    #     box='horizontal1',
    #     title='Top Keywords',
    #     content='Placeholder for now',
    # ))