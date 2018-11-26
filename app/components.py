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
# Components data
############################################################


def book_to_dropdown_item(book):
    return {'label': book['original_title'], 'value': int(book['book_id'])}


def books_to_dropdown(book_data):
    dropdown_items = list()
    for k in range(0, len(book_data)):
        dropdown_items.append(book_to_dropdown_item(book_data.iloc[k, :]))

    return dropdown_items


def _model_to_dropdown(cb_model_label):
    return {'label': cb_model_label, 'value': cb_model_label}


def models_to_dropdown(cb_models):
    return [_model_to_dropdown(cb_model_label)
            for cb_model_label, _ in cb_models.items()]

############################################################
# Rating form
############################################################


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
                    max=5,
                    value=0
                )
            ]
        )
    ]
)


############################################################
# Book rendering
############################################################


def get_rating_form(book_data):
    return html.Form(
        children=[
            book_selection(book_data),
            _RATING_SELECTION,
        ]
    )


def render_book(book_data, rating):
    book_layout = html.Div(
        className='col-sm-2 d-flex flex-column align-items-center',
        children=[
            html.Img(src=book_data['image_url']),
            html.Small(book_data['authors'], style={'color': '#999999'}),
            html.Strong(book_data['original_title']),
            html.Small(f'Rating: {rating}', style={'color': '#F9A602'}),
        ],
        style={'padding': '20', 'text-align': 'center'}
    )

    return book_layout


def rated_books_layout(book_data, book_ratings):
    rated_books = book_data[book_data.book_id.isin(book_ratings.keys())]

    layout = html.Div(
        className='d-flex flex-row',
        children=[render_book(book, book_ratings[str(book.book_id)])
                  for _, book in rated_books.iterrows()],
        style={
            'width': '100%',
            'margin': 10
        }
    )

    return layout
