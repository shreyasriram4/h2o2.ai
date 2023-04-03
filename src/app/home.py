from h2o_wave import main, app, Q, ui, on, handle_on, data
from src.app.helper import add_card, clear_cards
import pandas as pd


async def page1_upload(q: Q):
    q.page['sidebar'].value = '#home'

    # When routing, drop all the cards except of the main ones
    # (header, sidebar, meta).
    clear_cards(q)

    add_card(q, 'fileupload', ui.form_card(
            box=ui.box(zone='horizontal', height='135px',
                       width='100%'),
            items=[
                ui.file_upload(name='file_upload',
                               label="Upload reviews here (Recommended format"
                               + ": csv file with 2 columns 'Time' and" +
                               " 'Text')",
                               compact=True, multiple=False,
                               file_extensions=['csv']),
                ui.button(name='form_submit', label='Submit', primary=True)
            ]
        ))
    await q.page.save()


async def page1_preview(q: Q, df):
    q.page['sidebar'].value = '#home'
    clear_cards(q)
    # When routing, drop all the cards except of the main ones
    # (header,sidebar,meta).
    add_card(q, 'fileupload', ui.form_card(
        box=ui.box(zone='horizontal', height='132px',
                   width='100%'),
        items=[
            ui.file_upload(name='file_upload', label='Upload reviews',
                           compact=True, multiple=False,
                           file_extensions=['csv']),
            ui.button(name='form_submit', label='Submit', primary=True)
        ]
    ))

    add_card(q, 'datapreview', ui.form_card(
        box=ui.box(
                zone='horizontal1', height='620px',
                width='100%'),
        title='Preview top 30 entries',
        items=[ui.table(
                    name='preview',
                    columns=[ui.table_column(
                                name=x,
                                label=x,
                                sortable=True,
                                max_width='500px',
                                searchable=True,
                                cell_overflow='wrap'
                                ) for x in df.columns],
                    rows=[ui.table_row(
                            name=str(i),
                            cells=list(map(
                                    str,
                                    df[df.columns].values.tolist(
                                        )[i]))) for i in df.index[0:30]],
                    height='570px'), ]))
    await q.page.save()
