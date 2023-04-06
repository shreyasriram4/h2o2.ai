from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.helper import add_card, clear_cards
from src.visualisation.dashboard_viz import topics_pie_chart
from src.visualisation.dashboard_viz import visualise_all_topics
from src.visualisation.dashboard_viz import topics_bar_chart_over_time
from src.visualisation.dashboard_viz import extract_num_topics
from src.visualisation.dashboard_viz import html_output


@on('#topics')
async def page3(q: Q, df):
    q.page['sidebar'].value = '#topics'
    # When routing, drop all the cards except of the main ones
    # (header, sidebar, meta).
    clear_cards(q)

    total_num_topics = await extract_num_topics(df)
    add_card(q, 'article', ui.markdown_card(
        box='horizontal',
        title='',
        content='<div align="center"><h2>Exploring Topics </br></br>' +
        f'There are a total of {total_num_topics} topics </h2></div>'
    ))

    # add_card(q, 'linechart', ui.frame_card(
    #     box=ui.box(zone='horizontal1', size='2'),
    #     title="",
    #     content=await topics_line_chart_by_quarter(df),
    #     ))

    add_card(q, 'top_keywords', ui.frame_card(
        box=ui.box(zone='horizontal1', size='2'),
        title="Top Words From Each Topic",
        content=await html_output(visualise_all_topics(df)),
        ))

    add_card(q, 'topic_piechart', ui.frame_card(
        box=ui.box(zone='horizontal2', size='1'),
        title="Frequency of topics",
        content=await html_output(topics_pie_chart(df)),
        ))

    add_card(q, 'barchart2', ui.frame_card(
        box=ui.box(zone='horizontal2', size='2'),
        title="Topics over time",
        content=await html_output(
            topics_bar_chart_over_time(df, time_frame='Q')),
        ))

    await q.page.save()
