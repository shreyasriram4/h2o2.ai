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

    add_card(q, 'searchbar2', ui.form_card(
        box='horizontal1', 
        items=[
            ui.textbox(name='textbox_placeholder', label='Input topic label:', placeholder="Enter 'Food' if you would like to explore more about this topic"),
            ui.button(name='show_inputs', label='Submit', primary=True),],
            ),)
    
    add_card(q, 'notable_reviews', ui.tall_info_card(box='horizontal2', name='', title='Speed',
                                                    caption='The models are performant thanks to...', icon='SpeedHigh'))
    
    await q.page.save()