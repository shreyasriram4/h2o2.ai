from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.helper import add_card, clear_cards
from src.visualisation.dashboard_viz import reformat_data
from src.visualisation.dashboard_viz import topics_line_chart_by_quarter
from src.visualisation.dashboard_viz import topics_pie_chart
from src.visualisation.dashboard_viz import visualise_all_topics
from src.visualisation.dashboard_viz import topics_bar_chart_over_time
from src.visualisation.dashboard_viz import extract_num_topics

import pandas as pd

df = pd.read_csv('data/predicted/reviews.csv')
df = reformat_data(df)


@on('#topics')
async def page3(q: Q):
    q.page['sidebar'].value = '#topics'
    # When routing, drop all the cards except of the main ones
    # (header, sidebar, meta).
    clear_cards(q)

    total_num_topics = await extract_num_topics(df)
    add_card(q, 'article', ui.markdown_card(
        box='horizontal',
        title='',
        content=f'<div align="center"><h2>Exploring Topics </br></br>' +
        'There are a total of {total_num_topics} topics </h2></div>'
    ))

    # add_card(q, 'numbers', ui.large_stat_card(
    #     box=ui.box(zone='horizontal1', size='1'),
    #     title="",
    #     value=await extract_num_topics(df),
    #     aux_value='Topics',
    #     caption=''
    #     ))

    add_card(q, 'linechart', ui.frame_card(
        box=ui.box(zone='horizontal1', size='2'),
        title="",
        content=await topics_line_chart_by_quarter(df),
        ))

    add_card(q, 'barchart2', ui.frame_card(
        box=ui.box(zone='horizontal1', size='2'),
        title="",
        # content=await topics_bar_chart_by_quarter(df),
        content=await topics_bar_chart_over_time(df, time_frame='Q')
        ))

    add_card(q, 'topic_piechart', ui.frame_card(
        box=ui.box(zone='horizontal2', size='1'),
        title="",
        content=await topics_pie_chart(df),
        ))

    add_card(q, 'top_keywords', ui.frame_card(
        box=ui.box(zone='horizontal2', size='2'),
        title="Top Words",
        content=await visualise_all_topics(df),
        ))

    await q.page.save()
