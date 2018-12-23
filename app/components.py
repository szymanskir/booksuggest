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


def get_goodreads_url(goodreads_id):
    return f'{resources.GOODREADS_URL}/{goodreads_id}'


def get_book_ratings(user_data, selected_user_id):
    user_ratings = resources.USER_DATA[
        resources.USER_DATA['user_id'] == selected_user_id
    ].sort_values(by='rating', ascending=False)

    book_ratings = {row['book_id']: row['rating']
                    for _, row in user_ratings.iterrows()}

    return book_ratings


def books_to_dropdown(book_data):
    """Converts book data to the dash dropdown values format

    The labels in the dropdown are book titles and the values
    are the ids from the dataset.
    """
    return [{'label': row['title'], 'value': idx}
            for idx, row in book_data.iterrows()]


def models_to_dropdown(cb_models):
    """Converts available models to the dash dropdown values format

    Both labels and values in the dropdown are model names.
    """
    return [{'label': cb_model_label, 'value': cb_model_label}
            for cb_model_label in cb_models.keys()]


def users_to_dropdown(user_data):
    """Converts user ratings to dash dropdown values format
    """
    user_ids = user_data['user_id'].unique()
    return [{'label': f'user {idx}', 'value': idx}
            for idx in user_ids]

############################################################
# Book rendering
############################################################


def render_book(book_data, extra_html):
    """Creates an html representation of a book.
    """
    extra_html = extra_html if extra_html else html.Div()

    book_layout = html.Div(
        className='col-sm-2 d-flex flex-column align-items-center',
        children=[
            html.A(
                href=get_goodreads_url(book_data['goodreads_book_id']),
                children=[
                    html.Img(src=book_data['image_url']),
                ]
            ),
            html.Small(book_data['authors'], style={'color': '#999999'}),
            html.Strong(book_data['original_title']),
            extra_html
        ],
        style={'padding': '20', 'text-align': 'center'}
    )

    return book_layout


def create_books_layout(
        book_data,
        selected_books,
        extra_html_label,
        extra_html_color
):
    """Creates an html layout composed of specfied books.
    """
    layout = html.Div(
        className='d-flex flex-row flex-wrap',
        children=[
            render_book(
                book_data.loc[book_id],
                _create_extra_html(
                    extra_html_label,
                    selected_books[book_id],
                    extra_html_color
                )
            )
            for book_id in selected_books
        ],
        style={
            'max-height': 500
        }
    )

    return layout


def _create_extra_html(label, value, color):
    """Creates extra html elements for book rendering.
    """
    value = round(value, 2)
    return html.Small(
        f'{label}: {value}',
        style={'color': color}
    )
