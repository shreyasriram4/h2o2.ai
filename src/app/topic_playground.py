from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.helper import add_card, clear_cards
from src.visualisation.dashboard_viz import get_subtopics
from src.visualisation.dashboard_viz import sentiment_pie_chart
from src.visualisation.dashboard_viz import extract_top_reviews
<<<<<<< HEAD
=======
from src.visualisation.dashboard_viz import html_output
>>>>>>> main
import pandas as pd

# @on('#topic_playground')


async def page4_input(q: Q, df):
    q.page['sidebar'].value = '#topic_playground'
    clear_cards(q)
    # When routing, drop all the cards except of the main ones
    # (header, sidebar, meta).

    add_card(q, 'dropdown_topics', ui.form_card(
        box=ui.box(zone='horizontal', height='132px', width='100%'),
        items=[ui.dropdown(name='playground_topic',
                           label='Explore a specific topic label',
                           placeholder='Select the specific topic ' +
                           'from the dropdown list',
                           choices=[
                            ui.choice(name=x, label=x)
                            for x in df.topic.unique()]),
                ui.button(name='playground_submit', label='Submit',
                          primary=True)]
    ))

    await q.page.save()


async def page4_result(q: Q, topics, df):
    q.page['sidebar'].value = '#topic_playground'
    clear_cards(q)
    # When routing, drop all the cards except of the main ones
    # (header, sidebar, meta).

    add_card(q, 'dropdown_topics', ui.form_card(
        box=ui.box(zone='horizontal', height='131px', width='100%'),
        items=[ui.dropdown(name='playground_topic',
                           label='Explore a specific topic label',
                           choices=[
                            ui.choice(name=x, label=x)
                            for x in df.topic.unique()]),
                ui.button(name='playground_submit', label='Submit',
                          primary=True)]
    ))
    extracted_pos = await extract_top_reviews(df, topics, 'positive')
    add_card(q, 'topic_data_pos_preview', ui.form_card(
        box=ui.box(zone='side1', order='1'),
        title='Top 50 positive reviews from specific topic: ' + topics,
        items=[ui.table(
            name='preview',
            columns=[
                ui.table_column(name='Reviews',
                                label='Reviews',
                                sortable=True,
                                max_width='600px',
                                searchable=True,
                                cell_overflow='wrap'),
                ui.table_column(name='tag',
                                label='Predicted Sentiment',
                                min_width='140px',
                                cell_type=ui.tag_table_cell_type(
                                                        name='tags',
                                                        tags=[ui.tag(
                                                            label='POSITIVE',
                                                            color='$mint')]
                                                                )
                                )
                ],
            rows=[
                ui.table_row(name=str(i),
                             cells=[i, 'POSITIVE']
                             ) for i in extracted_pos[0:50]],
            height='265px'),
            ]
        ))
    extracted_neg = await extract_top_reviews(df, topics, 'negative')
    add_card(q, 'topic_data_neg_preview', ui.form_card(
        box=ui.box(zone='side1', order='2'),
        title='Top 50 negative reviews from specific topic: ' + topics,
        items=[ui.table(
            name='preview',
            columns=[
                ui.table_column(name='Reviews',
                                label='Reviews',
                                sortable=True,
                                max_width='600px',
                                searchable=True,
                                cell_overflow='wrap'),
                ui.table_column(name='tag',
                                label='Predicted Sentiment',
                                min_width='140px',
                                cell_type=ui.tag_table_cell_type(
                                                        name='tags',
                                                        tags=[ui.tag(
                                                            label='NEGATIVE',
                                                            color='$red')]
                                                            )
                                )
                ],
            rows=[
                ui.table_row(name=str(i),
                             cells=[i, 'NEGATIVE']
                             ) for i in extracted_neg[0:50]],
            height='265px'),
            ]
        ))

    add_card(q, 'Subtopic', ui.frame_card(
        box=ui.box(zone='side2', order='1'),
        title=f"Subtopics from specific topic: {topics}",
        content=await html_output(get_subtopics(df, topics))),
        )

    add_card(q, 'sentimentbreakdown_playground', ui.frame_card(
        box=ui.box(zone='side2', order='2'),
<<<<<<< HEAD
        title=f"Sentiment Breakdown from specific topic: {topics}",
        content=await sentiment_pie_chart(data=df[df["topic"] == topics]),
=======
        title="Sentiment Breakdown from specific topic: " + topics,
        content=await html_output(sentiment_pie_chart(
            data=df[df["topic"] == topics])),
>>>>>>> main
        ))
    await q.page.save()
