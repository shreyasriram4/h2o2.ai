from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.helper import add_card, clear_cards
# from helper import add_card, clear_cards
import pandas as pd
from src.visualisation.dashboard_viz import reformat_data, sentiment_pie_chart, sentiment_line_chart_over_time
# from visualisation.dashboard_viz import reformat_data, sentiment_pie_chart, sentiment_line_chart_over_time

df = pd.read_csv('data/processed/reviews.csv')
df = reformat_data(df)

@on('#home')
async def page1(q: Q):
    q.page['sidebar'].value = '#home'
    clear_cards(q)  # When routing, drop all the cards except of the main ones (header, sidebar, meta).

    # add_card(q, 'article', ui.markdown_card(
    #     box='horizontal',
    #     title='',
    #     content= '<div align="center"><h2>Start exploring your data!</h2></div>',
        
    # ))

    add_card(q, 'fileupload', ui.form_card(
            box='horizontal',
            items=[
                ui.file_upload(name='file_upload', label='Upload reviews', compact=True,
                               multiple=True, file_extensions=['csv']),
                ui.button(name='submit', label='Submit', primary=True)
            ]
        ))
    
    # add_card(q, 'searchbar2', ui.form_card(
    #     box='horizontal', 
    #     items=[
    #         ui.picker(name='picker', label='Input topic labels:', choices=[
    #             ui.choice(name='people', label='People'),
    #             ui.choice(name='staff', label='Staff'),
    #             ui.choice(name='location', label='Location'),
    #         ], values=['people']),
    #         ui.button(name='show_inputs', label='Submit', primary=True),]))

    add_card(q, 'number_reviews', ui.small_stat_card(
        box='50 50 5 5', 
        title='',
        value=f'Generating Total Number of Reviews: {df.shape[0]}'),)
        # value=[ui.text_l(f'**Generating Total Number of Reviews**: {df.shape[0]}')]))
    
    add_card(q, 'data_preview', ui.form_card(
        box='horizontal1', 
        items=[ui.table(
            name='preview',
            columns= 
                [ui.table_column(name=x, 
                                 label=x,
                                 sortable=True,
                                 cell_overflow='wrap') for x in ['date', 'partially_cleaned_text']],
            rows = [ui.table_row(
                name = str(i),
                cells = list(map(str, df[['date', 'partially_cleaned_text']].values.tolist()[i]))
                ) for i in df.index[0:29]])
            ]))
    
    # add_card(q, 'number_reviews', ui.markdown_card(
    #     box='horizontal1', 
    #     title='',
    #     content=f'**Generating Total Number of Reviews**: {df.shape[0]}'),)

    await q.page.save()