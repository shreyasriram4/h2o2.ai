from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.home import page1_upload, page1_preview
from src.app.sentiments import page2
from src.app.topics import page3
from src.app.topic_playground import page4_input, page4_result

import pandas as pd
from src.visualisation.dashboard_viz import reformat_data


async def init(q: Q) -> None:
    q.page['meta'] = ui.meta_card(
        box='', layouts=[ui.layout(
            breakpoint='xs', min_height='100vh',
            zones=[ui.zone(
                'main', size='1',
                direction=ui.ZoneDirection.COLUMN,
                zones=[
                    # Segment out the header
                    ui.zone('header', size='60px'),
                    # Segment out sidebar and content
                    ui.zone('body',
                            size='1', direction=ui.ZoneDirection.ROW,
                            zones=[
                                ui.zone('sidebar', size='200px'),
                                ui.zone('content', zones=[
                                    # Specify various zones and use the
                                    # one that is currently needed.
                                    # Empty zones are ignored.
                                    ui.zone(
                                        'horizontal',
                                        direction=ui.ZoneDirection.ROW,
                                        size='18%'),
                                    ui.zone(
                                        'vertical2',
                                        direction=ui.ZoneDirection.ROW,
                                        zones=[ui.zone(
                                            'side1',
                                            direction=ui.ZoneDirection.COLUMN,
                                            size='60%'),
                                            ui.zone(
                                            'side2',
                                            direction=ui.ZoneDirection.COLUMN,
                                            size='40%')
                                        ]),
                                    ui.zone(
                                        'horizontal1',
                                        direction=ui.ZoneDirection.ROW,
                                        size='41%'),
                                    ui.zone(
                                        'horizontal2',
                                        direction=ui.ZoneDirection.ROW,
                                        size='41%'),])]),
                            ])])])

    q.page['sidebar'] = ui.nav_card(
        box='sidebar', color='card', title='H2O2.ai', subtitle="NUS DSA4263",
        value=f'#{q.args["#"]}' if q.args['#'] else '#home',
        image='https://freesvg.org/img/1531595967.png', items=[
            ui.nav_group('Menu', items=[
                ui.nav_item(name='#home', label='Home', icon='Home'),
                ui.nav_item(name='#sentiments', label='Sentiment',
                            icon='SentimentAnalysis'),
                ui.nav_item(name='#topics', label='Topics',
                            icon='StackedLineChart'),
                ui.nav_item(name='#topic_playground', label='Topic Playground',
                            icon='Sunny')
            ]),
        ])
    q.page['header'] = ui.header_card(
        box='header', title='Voice of Customer', subtitle='',
        icon='Microphone', color='primary'
    )


@app('/')
async def serve(q: Q):
    # Run only once per client connection.
    if not q.client.initialized:
        q.client.cards = set()
        await init(q)
        q.client.initialized = True

    # # psuedo code for integration
    # page1_preview will download the test.csv
    # if test.csv exist:
    #     save df of predict(test.csv) as predict_test_df
    #     save df of topic_modelling(test.csv) as topic_modelling_test_df
    #     merge both df and chose the columns we want
    # thereafter, our page2-page4 functions will take in these df according

    # can keep this for default (training data)
    input_df = pd.read_csv('data/processed/reviews.csv')
    input_df = reformat_data(input_df)

    df = pd.read_csv('data/predicted/reviews.csv')
    df = reformat_data(df)

    route = q.args['#']
    if route == 'home' or route is None:
        if q.args.file_upload:
            files = q.args.file_upload
            await page1_preview(q, files, input_df)
        else:
            await page1_upload(q, input_df)
    elif route == 'sentiments':
        await page2(q, df)
    elif route == 'topics':
        await page3(q, df)
    elif route == 'topic_playground':
        if q.args.playground_topic:
            topics = q.args.playground_topic
            await page4_result(q, topics, df)
        else:
            await page4_input(q, df)

    # Handle routing.
    # await handle_on(q)
