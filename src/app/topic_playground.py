from h2o_wave import main, app, Q, ui, on, handle_on, data
from helper import add_card, clear_cards
    

@on('#topic_playground')
async def page4(q: Q):
    q.page['sidebar'].value = '#topic_playground'
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    add_card(q, 'article', ui.markdown_card(
        box='horizontal',
        title='',
        content= '<div align="center"><h2>Start exploring specfic topic!</h2></div>',
        
    ))
    await q.page.save()