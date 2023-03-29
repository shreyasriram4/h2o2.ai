from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.helper import add_card, clear_cards
from src.visualisation.dashboard_viz import extract_top_topic_reviews
from src.visualisation.dashboard_viz import get_subtopics
from src.visualisation.dashboard_viz import sentiment_pie_chart
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
    extracted = await extract_top_topic_reviews(df, topics)
    print(extracted)
    add_card(q, 'topic_data_reviews', ui.form_card(
        # box=ui.box(zone='side1', size='1', width='50%'),
        box=ui.box(zone='side1'),
        title='Notable reviews from specific topic: ' + topics,
        items=[ui.table(
            name='preview',
            columns=[
                ui.table_column(name='Reviews',
                                label='Reviews',
                                sortable=True,
                                max_width='500px',
                                searchable=True,
                                cell_overflow='wrap'),
                ui.table_column(name='tag',
                                label='Predicted Sentiment',
                                min_width='140px',
                                max_width='180px',
                                sortable=True,
                                filterable=True,
                                cell_type=ui.tag_table_cell_type(
                                                    name='tags',
                                                    tags=[ui.tag(
                                                        label='positive',
                                                        color='$mint'),
                                                        ui.tag(
                                                        label='negative',
                                                        color='$red')])
                                )
                    ],
            rows=[
                ui.table_row(name=str(i),
                             cells=list(map(
                                    str,
                                    df[['partially_cleaned_text',
                                        'sentiment'
                                        ]].values.tolist()[i]
                                        ))) for i in extracted.index[:100]],
            height='550px',
            ),
            ]
        ))

    add_card(q, 'Subtopic', ui.frame_card(
        # box=ui.box(zone='horizontal1', size='1'),
        box=ui.box(zone='side2', order='1'),
        title=f"Subtopics from specific topic: {topics}",
        content=await get_subtopics(df, topics)),
        )

    add_card(q, 'sentimentbreakdown_playground', ui.frame_card(
        # box=ui.box(zone='horizontal2', size='1', width='12%'),
        box=ui.box(zone='side2', order='2'),
        title="Sentiment Breakdown from specific topic: " + topics,
        content=await sentiment_pie_chart(data=df[df["topic"] == topics]),
        ))
    await q.page.save()
