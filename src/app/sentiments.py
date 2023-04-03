from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.helper import add_card, clear_cards

from src.visualisation.dashboard_viz import sentiment_pie_chart
from src.visualisation.dashboard_viz import sentiment_line_chart_over_time
from src.visualisation.dashboard_viz import topics_bar_chart


@on('#sentiments')
async def page2(q: Q, df):
    q.page['sidebar'].value = '#sentiments'
    # When routing, drop all the cards except of the main ones
    # (header, sidebar, meta).
    clear_cards(q)
    print(df)
    add_card(q, 'article', ui.markdown_card(
        box='horizontal',
        title='',
        content='<div align="center"><h2>Exploring Sentiments</h2></div>',
    ))

    add_card(q, 'piechart1', ui.frame_card(
        box=ui.box(zone='horizontal1', size='1'),
        title='Overall Sentiment Breakdown',
        content=await sentiment_pie_chart(data=df),
        ))

    add_card(q, 'chart2', ui.frame_card(
        box=ui.box(zone='horizontal1', size='2'),
        title='Sentiments over Time',
        content=await sentiment_line_chart_over_time(data=df),
    ))

    add_card(q, 'chart3', ui.frame_card(
        box=ui.box(zone='horizontal2', size='4'),
        title='Topics by Sentiment',
        content=await topics_bar_chart(data=df),
        ))

    # pos_freq_words = await extract_most_freq_words(data=df, positive=True)
    # add_card(q, 'topic_data_pos_preview', ui.form_card(
    #     box=ui.box(zone='horizontal2', size='1', order='1', width='150px'),
    #     title='Most frequent words appearing in Positive Reviews',
    #     items=[ui.table(
    #         name='preview',
    #         columns=[
    #                 ui.table_column(name='Word',
    #                                 label='Word',
    #                                 sortable=True,
    #                                 cell_overflow='wrap'),
    #                 ],
    #         rows=[
    #             ui.table_row(name=str(i),
    #                          cells=[str(i)]) for i in pos_freq_words[0:10]],
    #         width='200px'),
    #         ]
    #     ))
    # neg_freq_words = await extract_most_freq_words(data=df, positive=False)
    # add_card(q, 'topic_data_neg_preview', ui.form_card(
    #     box=ui.box(zone='horizontal2', size='1', order='2', width='150px'),
    #     title='Most frequent words appearing in Negative Reviews',
    #     items=[ui.table(
    #         name='preview',
    #         columns=[
    #                 ui.table_column(name='Word',
    #                                 label='Word',
    #                                 sortable=True,
    #                                 cell_overflow='wrap'),
    #                 ],
    #         rows=[ui.table_row(name=str(i),
    #                            cells=[str(i)])
    #              for i in neg_freq_words[0:10]],
    #         width='200px'),
    #         ]
    #     ))

    await q.page.save()
