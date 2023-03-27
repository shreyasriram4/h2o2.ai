from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.helper import add_card, clear_cards
from src.visualisation.dashboard_viz import reformat_data, extract_top_reviews
from src.visualisation.dashboard_viz import visualise_all_topics_playground
from src.visualisation.dashboard_viz import sentiment_pie_chart
import pandas as pd

# @on('#topic_playground')


async def page4_input(q: Q):
    q.page['sidebar'].value = '#topic_playground'
    clear_cards(q)
    # When routing, drop all the cards except of the main ones
    # (header, sidebar, meta).

    add_card(q, 'inserttopic', ui.form_card(
        box=ui.box(zone='horizontal', height='132px', width='100%'),
        items=[
            ui.textbox(name='playground_topic',
                       label='Explore a specific topic label:',
                       placeholder="Enter 'Baked Goods' if you would like "
                       + "to explore more about this topic"),
            ui.button(name='playground_submit', label='Submit', primary=True)
            ]
        ))
    await q.page.save()


async def page4_result(q: Q, topics):
    q.page['sidebar'].value = '#topic_playground'
    clear_cards(q)
    # When routing, drop all the cards except of the main ones
    # (header, sidebar, meta).

    add_card(q, 'inserttopic', ui.form_card(
        box=ui.box(zone='horizontal', height='132px', width='100%'),
        items=[
            ui.textbox(name='playground_topic',
                       label='Explore a specific topic label:',
                       placeholder="Enter 'Baked Goods' if you would like "
                       + "to explore more about this topic"),
            ui.button(name='playground_submit', label='Submit', primary=True)]
        ))

    df = pd.read_csv('data/predicted/reviews.csv')
    df = reformat_data(df)

    # print what users keyed in
    print('look here', topics)

    # search for the positive and negative reviews based on specific topics
    extracted_neg = await extract_top_reviews(df, topics, 'negative')
    extracted_pos = await extract_top_reviews(df, topics, 'positive')
    print('done')

    add_card(q, 'topic_data_pos_preview', ui.form_card(
        box=ui.box(zone='horizontal1', size='1'),
        title='Top 10 positive notable reviews from specific topic: ' + topics,
        items=[ui.table(
            name='preview',
            columns=[
                ui.table_column(name='Reviews',
                                label='Reviews',
                                sortable=True,
                                max_width='420px',
                                # searchable = True,
                                cell_overflow='wrap'),
                ui.table_column(name='tag',
                                label='Predicted Sentiment',
                                min_width='140px',
                                cell_type=ui.tag_table_cell_type(
                                                    name='tags',
                                                    tags=[ui.tag(
                                                        label='POSITIVE',
                                                        color='$mint')])
                                )
                    ],
            rows=[
                ui.table_row(name=str(i),
                             cells=[i, 'POSITIVE']
                             ) for i in extracted_pos[0:10]],
            height='265px'),
            ]
        ))

    add_card(q, 'N-gram', ui.frame_card(
        box=ui.box(zone='horizontal1', size='2'),
        title="Top words in 1-gram",
        content=await visualise_all_topics_playground(data=df, topic=[topics]),
        ))

    add_card(q, 'topic_data_neg_preview', ui.form_card(
        box=ui.box(zone='horizontal2', size='1'),
        title='Top 10 negative notable reviews from specific topic: ' + topics,
        items=[ui.table(
            name='preview',
            columns=[
                ui.table_column(name='Reviews',
                                label='Reviews',
                                sortable=True,
                                max_width='420px',
                                # searchable = True,
                                cell_overflow='wrap'),
                ui.table_column(name='tag',
                                label='Predicted Sentiment',
                                min_width='140px',
                                cell_type=ui.tag_table_cell_type(
                                                        name='tags',
                                                        tags=[ui.tag(
                                                            label='NEGATIVE',
                                                            color='$red')])
                                )],
            rows=[
                ui.table_row(name=str(i),
                             cells=[i, 'NEGATIVE']
                             ) for i in extracted_neg[0:10]],
            height='265px'),
            ]
        ))

    add_card(q, 'sentimentbreakdown_playground', ui.frame_card(
        box=ui.box(zone='horizontal2', size='2', width='12%'),
        title="Overall Sentiment Breakdown from specific topic: " + topics,
        content=await sentiment_pie_chart(data=df[df["topic"] == topics]),
        ))
    await q.page.save()
