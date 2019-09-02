![Travis](https://travis-ci.com/szymanskir/Recommendation-system.svg?token=vMgapB9HzV6RFvox4Fiq&branch=master)

# Recommendation System for Books

Bachelor of Engineering thesis by Paweł Rzepiński and Ryszard Szymański under the supervision of Agnieszka Jastrzębska, Ph.D. Eng. The objective was to develop a recommender system for books using novel dataset. Both collaborative-filtering and content-based approaches were considered. Implemented recommendation models were accessible using web application allowing users to explore the dataset and compare results for both approaches: "Similar books to X" panel presenting items similar to the selected book and "You may also like X, Y, Z" containing recommendations based on books rated by the selected user.

![content-based methods](https://github.com/szymanskir/booksuggest/blob/master/docs/cb-screen.png "Content-based methods")

![collaborative filtering methods](https://github.com/szymanskir/booksuggest/blob/master/docs/cf-screen.png "Collaborative filtering methods")

Full showcase video available at [Google Drive](https://drive.google.com/open?id=1Se6Xu496xKTsYOAvgedjifapsvz2zaTD).

Thesis folder contains both [thesis](thesis/thesis.pdf) and [abstract](thesis/abstract-EN.pdf).

## Documentation

Documentation of the recommendation module can be found in the `docs` folder. Main page is located at `docs/_build/html/index.html`.

## Project structure

    ├── Makefile           <- Makefile with commands like `make data`, `make models`, `make scores`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Codebase documentation.
    │
    ├── models             <- Trained and serialized models, model predictions.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1-rzepinskip-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as PDF files.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt        <- The requirements file for the web application.
    ├── requirements-dev.txt    <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- Project's main module. Can be installed with pip command.
    └── booksuggest        <- Source code of the recommendation module.
        │
        ├── data           <- Scripts to download or generate data.
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │                   predictions.
        │
        └── evaluation     <- Scripts to evaluate scores and validate results against ground-truth data.

## Setup instructions

All commands mentioned below should be run from the project's root folder. Use `make help` to display help information about available commands.

### Prerequisites

- UNIX based system
- GNU Make
- Python 3.7
- pip

### Viewing results

1. Create a virtual environment:
    ```bash
    make create_environment
    ```
2. Activate the virtual environment:
    ```bash
    source rs-venv/bin/activate
    ```
3. Install packages required by the web application:
    ```bash
    make app_requirements
    ```
4. Run the app:
    ```bash
    make app
    ```
5. Enter the web page address displayed in the console. Web application should be accessible at <http://127.0.0.1:8050/>.

### Reproducing the analysis

1. Create a virtual environment:
    ```bash
    make create_environment
    ```
2. Activate the virtual environment:
    ```bash
    source rs-venv/bin/activate
    ```
3. Install packages required for development:
    ```bash
    make requirements
    ```
4. Download the raw data:
    ```bash
    make data
    ```
5. Train models:
    ```bash
    make models
    ```
6. Evaluate models:
    ```bash
    make scores
    ```

Comments:

- When using the whole dataset the `make models` command takes about 20 minutes, `make scores` lasts more than 12h.
- To check pipeline on the small subset of data use `TEST_RUN=1` parameter when running make commands. Then, the whole process should take about 5 minutes. Example: `make scores TEST_RUN=1`
- To utilize make's parallelization use `-j <n_jobs>` parameter where `<n_jobs>` specifies the number of parallel jobs run. Most often, `n_jobs` should be equal to the number of cores in the processor, although there are also some RAM requirements when using whole dataset. Example: `make scores -j 2`

## Acknowledgments

- Recommendation methods mostly from [surprise library](https://github.com/NicolasHug/Surprise).
- Project structure based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/).
- Thesis based on [latex-mimosis](https://github.com/rzepinskip/latex-mimosis) template by [Bastian Rieck](https://bastian.rieck.me/).
