from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.helper import add_card, clear_cards
    

@on('#topics')
async def page3(q: Q):
    q.page['sidebar'].value = '#topics'
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    
    add_card(q, 'article', ui.markdown_card(
        box='horizontal',
        title='',
        content= '<div align="center"><h2>Overall topics</h2></div>',
        
    ))
    await q.page.save()