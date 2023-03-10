from h2o_wave import main, app, Q, ui, on, handle_on, data
from helper import add_card, clear_cards


@on('#home')
async def page1(q: Q):
    q.page['sidebar'].value = '#home'
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    add_card(q, 'article', ui.markdown_card(
        box='horizontal',
        title='',
        content= '<div align="center"><h2>Start exploring your data!</h2></div>',
        
    ))
    for i in range(3):
        add_card(q, f'info{i}', ui.tall_info_card(box='horizontal1', name='', title='Speed',
                                                  caption='The models are performant thanks to...', icon='SpeedHigh'))

    await q.page.save()