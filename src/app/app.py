
from h2o_wave import main, app, Q, ui, on, handle_on, data
from home import page1
from sentiments import page2
from topics import page3
from topic_playground import page4

async def init(q: Q) -> None:
    q.page['meta'] = ui.meta_card(box='', layouts=[ui.layout(breakpoint='xs', min_height='100vh', zones=[
        ui.zone('main', size='1', direction=ui.ZoneDirection.COLUMN, zones=[
            # Segment out the header
            ui.zone('header', size='60px'),
            # Segment out sidebar and content
            ui.zone('body', size='1', direction=ui.ZoneDirection.ROW, zones=[
                ui.zone('sidebar', size='200px'),
                ui.zone('content', zones=[
                    # Specify various zones and use the one that is currently needed. Empty zones are ignored.
                    
                    ui.zone('horizontal', direction=ui.ZoneDirection.ROW),
                    ui.zone('horizontal1', direction=ui.ZoneDirection.ROW),
                    ui.zone('horizontal_sentiment', direction=ui.ZoneDirection.ROW, size = '100%'),
                    ui.zone('vertical_sentiment', direction=ui.ZoneDirection.COLUMN, size='100%'),
                    ui.zone('grid', direction=ui.ZoneDirection.ROW, wrap='stretch', justify='center')
                ]),
            ])
        ])
    ])])

    q.page['sidebar'] = ui.nav_card(
        box='sidebar', color='card', title='H2O2.ai', subtitle="NUS DSA4263",
        value=f'#{q.args["#"]}' if q.args['#'] else '#home',
        image='https://freesvg.org/img/1531595967.png', items=[
            ui.nav_group('Menu', items=[
                ui.nav_item(name='#home', label='Home', icon='Home'),
                ui.nav_item(name='#sentiments', label='Sentiment', icon='SentimentAnalysis'),
                ui.nav_item(name='#topics', label='Topics', icon='StackedLineChart'),
                ui.nav_item(name='#topic_playground', label='Topic Playground', icon='Sunny')
            ]),
        ])
    q.page['header'] = ui.header_card(
        box='header', title='Voice of Customer', subtitle='', icon = 'Microphone', color='primary'
    )

    # If no active hash present, render page1.
    if q.args['#'] is None:
        await page1(q)

@app('/')
async def serve(q: Q):
    # Run only once per client connection.
    if not q.client.initialized:
        q.client.cards = set()
        await init(q)
        q.client.initialized = True

    route = q.args['#']
    if route == 'home':
        await page1(q)
    elif route == 'sentiments':
        await page2(q)
    elif route == 'topics':
        await page3(q)
    elif route == 'topic_playground':
        await page4(q)


    # Handle routing.
    await handle_on(q)



