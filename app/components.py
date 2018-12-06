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


def books_to_dropdown(book_data):
    """Converts book data to the dash dropdown values format

    The labels in the dropdown are book titles and the values
    are the ids from the dataset.
    """
    return [{'label': row['original_title'], 'value': idx}
            for idx, row in book_data.iterrows()]


def models_to_dropdown(cb_models):
    """Converts available models to the dash dropdown values format

    Both labels and values in the dropdown are model names.
    """
    return [{'label': cb_model_label, 'value': cb_model_label}
            for cb_model_label in cb_models.keys()]

############################################################
# Rating form
############################################################


def book_selection(book_data):
    """Creates an html form for selecting books to review.
    """
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
    """Creates an html rating form component.
    """
    return html.Form(
        children=[
            book_selection(book_data),
            _RATING_SELECTION,
        ]
    )


def render_book(book_data, rating=None):
    """Creates an html representation of a book.
    """
    html_rating = html.Small(
        f'Rating: {rating}',
        style={'color': '#F9A602'}
    ) if rating else html.Div()

    book_layout = html.Div(
        className='col-sm-2 d-flex flex-column align-items-center',
        children=[
            html.Img(src=book_data['image_url']),
            html.Small(book_data['authors'], style={'color': '#999999'}),
            html.Strong(book_data['original_title']),
            html_rating
        ],
        style={'padding': '20', 'text-align': 'center'}
    )

    return book_layout


def rated_books_layout(book_data, book_ratings):
    """Creates an html layout composed of reviewed books.
    """
    rated_books = book_data.loc[book_ratings.keys()]

    layout = html.Div(
        className='d-flex flex-row',
        children=[render_book(book, book_ratings[idx])
                  for idx, book in rated_books.iterrows()],
        style={
            'width': '100%',
            'margin': 10
        }
    )

    return layout


def recommended_books_layout(book_data, book_ids):
    """Creates an html layout composed of recommended books.
    """
    recommended_books = book_data.loc[book_ids]

    layout = html.Div(
        className='d-flex flex-row',
        children=[render_book(book)
                  for _, book in recommended_books.iterrows()],
        style={
            'width': '100%',
            'margin': 10
        }
    )

    return layout
