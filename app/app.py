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
                            html.Div(
                                className='col-4',
                                children=[
                                    dcc.Dropdown(
                                        id='user-selection',
                                        className='flex-fill',
                                        placeholder='Select user...',
                                        options=components.users_to_dropdown(
                                            resources.USER_DATA
                                        )
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        id='rated-books',
                        className='border mb-5',
                        style={
                            'overflow': 'auto',
                        },
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
                        id='recommended-books-cf',
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
                                            resources.BOOK_DATA
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
              [Input('user-selection', 'value')])
def display_reviewed_books(selected_user_id):
    """Displays reviewed book

    Based on the json data saved in a hidden a div
    a layout of reviewed books is created and displayed.
    """
    user_ratings = resources.USER_DATA[
        resources.USER_DATA['user_id'] == selected_user_id
    ].sort_values(by='rating', ascending=False)

    book_ratings = {row['book_id']: row['rating']
                    for _, row in user_ratings.iterrows()}
    return components.rated_books_layout(resources.BOOK_DATA, book_ratings)


@app.callback(Output('recommended-books-cf', 'children'),
              [Input('model-selection-cf', 'value'),
               Input('user-selection', 'value')])
def display_cf_recommendations(model, selected_user_id):
    """Displays recommendations that were obtained using
    collaborative filtering methods.

    Based on the rated books saved in a hidden div,
    recommendations are calculated using collaborative
    filtering methods and a layout of recommended books
    is created and displayed.
    """

    if model is None or selected_user_id is None:
        return html.Div()

    user_ratings = resources.USER_DATA[
        resources.USER_DATA['user_id'] == selected_user_id
    ].sort_values(by='rating', ascending=False)

    book_ratings = {row['book_id']: row['rating']
                    for _, row in user_ratings.iterrows()}

    recommended_books = resources.CF_MODELS[model].recommend(
        book_ratings
    ) if book_ratings and model else list()

    return components.recommended_books_layout(resources.BOOK_DATA,
                                               recommended_books)


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
        resources.BOOK_DATA, recommended_books
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
