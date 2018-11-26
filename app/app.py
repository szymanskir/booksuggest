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
                                className='col-4 d-flex flex-row align-items-center',
                                children=[
                                    html.I(className='fas fa-star px-3'),
                                    html.H4(id='cb-title', children='Similar to X')
                                ]
                            ),
                            html.Div(
                                className='col-6',
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
            html.Div(id='rated-books-data', style={'display': 'none'})
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
def render_reviewed_books(rated_books):
    rated_books = dict() if rated_books is None else json.loads(rated_books)
    return components.rated_books_layout(resources.DATA, rated_books)


@app.callback(Output('rated-books-data', 'children'),
              [Input('add-book-review', 'n_clicks')],
              [State('rated-books-data', 'children'),
               State('book-title', 'value'),
               State('book-rating', 'value')])
def add_book_review(n_clicks, rated_books, book_id, rating):
    if rating is None or book_id is None:
        return None

    rated_books = dict() if rated_books is None else json.loads(rated_books)
    rated_books[int(book_id)] = rating

    return json.dumps(rated_books)


@app.callback(Output('cb-title', 'children'),
              [Input('model-selection-cb', 'value')],
              [State('rated-books-data', 'children')])
def cb_recommendations(model, rated_books):
    if rated_books is None:
        book_title = 'X'
    else:
        rated_books_keys = list(json.loads(rated_books).keys())
        random_index = random.randint(0, len(rated_books_keys) - 1)
        book_title_index = int(rated_books_keys[random_index]) - 1
        book_title = resources.DATA.loc[book_title_index,
                                        'original_title']

    return f'Similar to {book_title}'


if __name__ == '__main__':
    app.run_server(debug=True)
