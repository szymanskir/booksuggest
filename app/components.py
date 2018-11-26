import dash_core_components as dcc
import dash_html_components as html

import resources


############################################################
# Page Header - navigation bar
############################################################

_PAGE_HEADER_CONTENT = html.Div(
    className='d-flex flex-fill',
    children=[
        html.Div(
            className='col-6 d-flex flex-row',
            children=[
                html.A(
                    className='navbar-brand flex-row',
                    children=[
                        html.I(className='fas fa-book px-3'),
                        'Book recommendation system'
                    ],
                    style={
                        'font-size': 34,
                        'font-weight': 'bold'
                    }
                )
            ]
        ),
        html.Div(
            id='credits',
            className='col-6 d-flex flex-row-reverse',
            children=[
                html.Img(
                    className='navbar-brand',
                    src=resources._MINI_LOGO,
                    style={
                        'height': '60px',
                        'width': '60px'
                    }
                ),
                html.Div(
                    className='navbar-brand d-flex flex-column px-3',
                    children=[
                        html.P('Paweł Rzepiński'),
                        html.P('Ryszard Szymański')
                    ]
                )
            ]
        ),
    ]
)


PAGE_HEADER = html.Nav(
    className='navbar navbar-light navbar-expand col-12 mb-5',
    style={
        'background-color': '#e6e6e6',
        'vertical-align': 'middle'
    },
    children=[
        _PAGE_HEADER_CONTENT
    ]
)

############################################################
# Book rating component
############################################################


def book_to_dropdown_item(book):
    return {'label': book['original_title'], 'value': int(book['book_id'])}


def books_to_dropdown(book_data):
    dropdown_items = list()
    for k in range(0, len(book_data)):
        dropdown_items.append(book_to_dropdown_item(book_data.iloc[k, :]))

    return dropdown_items


def book_selection(book_data):
    return html.Div(
        className='form-group row',
        children=[
            html.Label(
                htmlFor='book-title',
                className='col-1 col-form-label',
                children='Book title'
            ),
            html.Div(
                className='col-9',
                children=[
                    dcc.Dropdown(
                        id='book-title',
                        placeholder='Select book...',
                        options=books_to_dropdown(book_data)
                    )
                ]
            )
        ]
    )


_RATING_SELECTION = html.Div(
    className='form-group row',
    children=[
        html.Label(
            htmlFor='book-rating',
            className='col-1 col-form-label',
            children='Rating'
        ),
        html.Div(
            className='col-9',
            children=[
                dcc.Input(
                    id='book-rating',
                    type='number',
                    min=0,
                    max=5
                )
            ]
        )
    ]
)


def get_rating_form(book_data):
    return html.Form(
        children=[
            book_selection(book_data),
            _RATING_SELECTION,
        ]
    )


def render_book(book_data):
    book_layout = html.Div(
        className='two columns',
        children=html.Img(src=book_data['image_url']),
        style={'padding': '20', 'margin': '10'}
    )

    return book_layout


def rated_books_layout(book_data, rated_books):
    rated_books = book_data[book_data.book_id.isin(rated_books.keys())]

    layout = html.Div(
        className='d-flex flex-row',
        children=[render_book(book) for _, book in rated_books.iterrows()],
        style={
            'width': '100%'
        }
    )

    return layout
