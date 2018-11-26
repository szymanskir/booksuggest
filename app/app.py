import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

import components

data = pd.read_csv('assets/book.csv')


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
                        children=components.rated_books_layout(
                            data,
                            {x: x for x in range(1, 3)})
                    ),
                    components.get_rating_form(data),
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
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        id='recomended-books-cf',
                        className='border mb-5',
                        style={
                            'overflow': 'auto',
                        },
                        children=components.rated_books_layout(
                            data,
                            {x: x for x in range(44, 49)})
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
                                    html.H4('Similar to X')
                                ]
                            ),
                            html.Div(
                                className='col-6',
                                children=[
                                    dcc.Dropdown(
                                        id='model-selection-cb',
                                        className='flex-fill',
                                        placeholder='Select model...',
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        id='recomended-books-cb',
                        className='border mb-5',
                        style={
                            'overflow': 'auto',
                        },
                        children=components.rated_books_layout(
                            data,
                            {x: x for x in range(48, 58)})
                    )
                ]
            ),


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
app.config['suppress_callback_exceptions'] = True


if __name__ == '__main__':
    app.run_server(debug=True)
