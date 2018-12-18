import dash
import dash_core_components as dcc
import dash_html_components as html
import json
import random

import components
import resources

from dash.dependencies import Input, State, Output


def serve_layout():
    return html.Div(
        className='d-flex flex-column align-items-center',
        children=[
            components.PAGE_HEADER,
            html.Div(
                className='col-10 mt-5 pb-3',
                children=[
                    html.Div(
                        className='d-flex flex-row align-items-center',
                        style={
                            'background-color': '#e6e6e6',
                        },
                        children=[
                            html.Div(
                                className='col-4 d-flex flex-row align-items-center',
                                children=[
                                    html.I(className='fas fa-info-circle px-3'),
                                    html.H4('Rated books')
                                ]
                            ),
                        ]
                    ),
                    html.Div(
                        id='rated-books',
                        className='border mb-5',
                        style={
                            'overflow': 'auto',
                        },
                    ),
                    components.get_rating_form(resources.DATA),
                    html.Button(
                        'Submit',
                        id='add-book-review',
                    )
                ]
            ),
            html.Div(
                className='col-10 mt-5 pb-3',
                children=[
                    html.Div(
                        className='d-flex flex-row align-items-center',
                        style={
                            'background-color': '#e6e6e6',
                        },
                        children=[
                            html.Div(
                                className='col-4 d-flex flex-row align-items-center',
                                children=[
                                    html.I(className='fas fa-star px-3'),
                                    html.H4('Recommended books for you')
                                ]
                            ),
                            html.Div(
                                className='col-6',
                                children=[
                                    dcc.Dropdown(
                                        id='model-selection-cf',
                                        className='flex-fill',
                                        placeholder='Select model...',
                                        options=components.models_to_dropdown(
                                            resources.CF_MODELS
                                        )
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        id='recomended-books-cf',
                        className='border mb-5',
                        style={'overflow': 'auto'}
                    )
                ]
            ),
            html.Div(
                className='col-10 mt-5 pb-3',
                children=[
                    html.Div(
                        className='d-flex flex-row align-items-center',
                        style={
                            'background-color': '#e6e6e6',
                        },
                        children=[
                            html.Div(
                                className='col-2 d-flex flex-row align-items-center',
                                children=[
                                    html.I(className='fas fa-star px-3'),
                                    html.H4(id='cb-title', children='Similar books')
                                ]
                            ),
                            html.Div(
                                className='col-5',
                                children=[
                                    dcc.Dropdown(
                                        id='book-selection',
                                        className='flex-fill',
                                        placeholder='Select book...',
                                        options=components.books_to_dropdown(
                                            resources.DATA
                                        )
                                    )
                                ]
                            ),
                            html.Div(
                                className='col-5',
                                children=[
                                    dcc.Dropdown(
                                        id='model-selection-cb',
                                        className='flex-fill',
                                        placeholder='Select model...',
                                        options=components.models_to_dropdown(
                                            resources.CB_MODELS
                                        )
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        id='recomended-books-cb',
                        className='border mb-5',
                        style={'overflow': 'auto'}
                    )
                ]
            ),

            # Hidden div inside the app that stores ratings
            html.Div(id='rated-books-data', style={'display': 'none'}),

            # Hidden div storing the book selected for cb recommendations
            html.Div(id='cb-selected-book', style={'display': 'none'})
        ]
    )


external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
    'https://use.fontawesome.com/releases/v5.5.0/css/all.css',
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)

app.layout = serve_layout


@app.callback(Output('rated-books', 'children'),
              [Input('rated-books-data', 'children')])
def display_reviewed_books(rated_books_data):
    """Displays reviewed book

    Based on the json data saved in a hidden a div
    a layout of reviewed books is created and displayed.
    """
    rated_books = json2dict(rated_books_data)
    return components.rated_books_layout(resources.DATA, rated_books)


@app.callback(Output('rated-books-data', 'children'),
              [Input('add-book-review', 'n_clicks')],
              [State('rated-books-data', 'children'),
               State('book-title', 'value'),
               State('book-rating', 'value')])
def add_book_review(n_clicks, rated_books_data, book_id, rating):
    """Adds a book review

    Adds a book review to a dictionary stored in a hidden div
    and saves it back to that hidden div.
    """
    rated_books = json2dict(rated_books_data)

    if book_id is not None and rating is not None:
        rated_books[book_id] = rating

    return dict2json(rated_books)


@app.callback(Output('recomended-books-cf', 'children'),
              [Input('model-selection-cf', 'value')],
              [State('rated-books-data', 'children')])
def display_cf_recommendations(model, rated_books_data):
    """Displays recommendations that were obtained using
    collaborative filtering methods.

    Based on the rated books saved in a hidden div,
    recommendations are calculated using collaborative
    filtering methods and a layout of recommended books
    is created and displayed.
    """
    rated_books = json2dict(rated_books_data)

    recommended_books = resources.CF_MODELS[model].recommend(
        rated_books
    ) if rated_books and model else list()

    return components.recommended_books_layout(resources.DATA,
                                               recommended_books)


@app.callback(Output('cb-selected-book', 'children'),
              [Input('model-selection-cb', 'value')],
              [State('rated-books-data', 'children')])
def select_book_for_cb(model, rated_books_data):
    """Selects a specific book from all reviewed books.

    Based on the reviewed books saved in a hidden div
    a random book is selected for content based methods.

    E.g. Harry Potter is selected for 'Similar to Harry Potter'
    recommendations.
    """
    rated_books = json2dict(rated_books_data)
    random_index = random.choice(
        list(rated_books.keys())
    ) if rated_books else None

    return json.dumps(random_index)


@app.callback(Output('recomended-books-cb', 'children'),
              [Input('model-selection-cb', 'value'),
               Input('book-selection', 'value')])
def display_cb_recommendation(model, selected_book_id):
    """Displays recommendations that were obtained using
    content based methods.

    Based on the rated books saved in a hidden div,
    recommendations are calculated using content based
    methods and a layout of recommended books is created
    and displayed.
    """
    if model is None or selected_book_id is None:
        return html.Div()

    recommended_books = resources.CB_MODELS[model].recommend({
        selected_book_id: 5
    })

    return components.recommended_books_layout(
        resources.DATA, recommended_books
    )


def json2dict(json_data):
    """Converts json_data to dictionary

    Wrapper around json.loads in order to handle Nones as an empty
    dictionary.
    """
    return dict(json.loads(json_data)) if json_data else dict()


def dict2json(dictionary):
    """Converts dictionary to json

    Wrapper around json.dumps, because json.dumps does not
    conserve the type of keys. They get casted to str by default.
    """
    return json.dumps(list(dictionary.items()))


if __name__ == '__main__':
    app.run_server(debug=True)
