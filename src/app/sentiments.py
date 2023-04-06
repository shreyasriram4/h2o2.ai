from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.helper import add_card, clear_cards

from src.visualisation.dashboard_viz import sentiment_pie_chart
from src.visualisation.dashboard_viz import sentiment_line_chart_over_time
from src.visualisation.dashboard_viz import topics_bar_chart, html_output


@on('#sentiments')
async def page2(q: Q, df):
    q.page['sidebar'].value = '#sentiments'
    # When routing, drop all the cards except of the main ones
    # (header, sidebar, meta).
    clear_cards(q)
    add_card(q, 'article', ui.markdown_card(
        box='horizontal',
        title='',
        content='<div align="center"><h2>Exploring Sentiments</h2></div>',
    ))

    add_card(q, 'piechart1', ui.frame_card(
        box=ui.box(zone='horizontal1', size='1'),
        title='Overall Sentiment Breakdown',
        content=await html_output(sentiment_pie_chart(data=df)),
        ))

    add_card(q, 'chart2', ui.frame_card(
        box=ui.box(zone='horizontal1', size='2'),
        title='Sentiments over Time',
        content=await html_output(sentiment_line_chart_over_time(data=df)),
    ))

    add_card(q, 'chart3', ui.frame_card(
        box=ui.box(zone='horizontal2', size='4'),
        title='Topics by Sentiment',
        content=await html_output(topics_bar_chart(data=df)),
        ))

    await q.page.save()
