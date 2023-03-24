from h2o_wave import main, app, Q, ui, on, handle_on, data
from helper import add_card, clear_cards
import pandas as pd

def load_data():
    data = pd.read_csv('../../data/processed/reviews.csv')
    data.date = pd.to_datetime(data.date)
    data['year_month'] = data.date.dt.to_period('M')
    return data

df = load_data()

def sentiment_over_time_df(df):
    df['date'] = df['date'].dt.strftime("%Y-%m-%d")
    time_df = df.groupby(['date', 'sentiment']).count().reset_index()
    return time_df

time_df = sentiment_over_time_df(df)

@on('#sentiments')
async def page2(q: Q):
    q.page['sidebar'].value = '#sentiments'
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    pos = (df['sentiment']==1).sum()
    neg = (df['sentiment']==0).sum()
    total = pos+neg
    add_card(q, 'piechart', ui.wide_pie_stat_card(
        box='horizontal',
        title='Overall Sentiment',
        pies=[
            ui.pie(label='Positive', value=f'{round(pos/total * 100, 1)}%', fraction=pos/total, color='$green'),
            ui.pie(label='Negative', value=f'{round(neg/total * 100, 1)}%', fraction=neg/total, color='$red'),
        ]
    ))

    add_card(q, 'time_trend', ui.plot_card(
        box='horizontal',
        title='Sentiment Over Time',
        data=data(fields=time_df.columns.tolist(),rows = time_df.values.tolist()),
        plot = ui.plot(marks=[ui.mark(type='line',x='=date',y='=partially_cleaned_text', color='=sentiment', color_range='$green $red')])
    ))

    add_card(q, 'chart3', ui.markdown_card(
        box='horizontal1',
        title='Sentiment by Topic',
        content='Placeholder for now',
    ))

    add_card(q, 'chart4', ui.markdown_card(
        box='horizontal1',
        title='Top Keywords',
        content='Placeholder for now',
    ))

    # add_card(q, 'chart1', ui.plot_card(
    #     box='horizontal',
    #     title='Chart 1',
    #     data=data('category country product price', 10, rows=[
    #         ('G1', 'USA', 'P1', 124),
    #         ('G1', 'China', 'P2', 580),
    #         ('G1', 'USA', 'P3', 528),
    #         ('G1', 'China', 'P1', 361),
    #         ('G1', 'USA', 'P2', 228),
    #         ('G2', 'China', 'P3', 418),
    #         ('G2', 'USA', 'P1', 824),
    #         ('G2', 'China', 'P2', 539),
    #         ('G2', 'USA', 'P3', 712),
    #         ('G2', 'USA', 'P1', 213),
    #     ]),
    #     plot=ui.plot([ui.mark(type='interval', x='=product', y='=price', color='=country', stack='auto',
    #                           dodge='=category', y_min=0)])
    # ))
    # add_card(q, 'chart2', ui.plot_card(
    #     box='horizontal',
    #     title='Chart 2',
    #     data=data('date price', 10, rows=[
    #         ('2020-03-20', 124),
    #         ('2020-05-18', 580),
    #         ('2020-08-24', 528),
    #         ('2020-02-12', 361),
    #         ('2020-03-11', 228),
    #         ('2020-09-26', 418),
    #         ('2020-11-12', 824),
    #         ('2020-12-21', 539),
    #         ('2020-03-18', 712),
    #         ('2020-07-11', 213),
    #     ]),
    #     plot=ui.plot([ui.mark(type='line', x_scale='time', x='=date', y='=price', y_min=0)])
    # ))
    # add_card(q, 'table', ui.form_card(box='vertical', items=[ui.table(
    #     name='table',
    #     downloadable=True,
    #     resettable=True,
    #     groupable=True,
    #     columns=[
    #         ui.table_column(name='text', label='Process', searchable=True),
    #         ui.table_column(name='tag', label='Status', filterable=True, cell_type=ui.tag_table_cell_type(
    #             name='tags',
    #             tags=[
    #                 ui.tag(label='FAIL', color='$red'),
    #                 ui.tag(label='DONE', color='#D2E3F8', label_color='#053975'),
    #                 ui.tag(label='SUCCESS', color='$mint'),
    #             ]
    #         ))
    #     ],
    #     rows=[
    #         ui.table_row(name='row1', cells=['Process 1', 'FAIL']),
    #         ui.table_row(name='row2', cells=['Process 2', 'SUCCESS,DONE']),
    #         ui.table_row(name='row3', cells=['Process 3', 'DONE']),
    #         ui.table_row(name='row4', cells=['Process 4', 'FAIL']),
    #         ui.table_row(name='row5', cells=['Process 5', 'SUCCESS,DONE']),
    #         ui.table_row(name='row6', cells=['Process 6', 'DONE']),
    #     ])
    # ]))

    await q.page.save()