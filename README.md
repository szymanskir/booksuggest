![Travis](https://travis-ci.com/szymanskir/Recommendation-system.svg?token=vMgapB9HzV6RFvox4Fiq&branch=master)

# Recommendation System for Books

Bachelor of Engineering thesis by Paweł Rzepiński and Ryszard Szymański.

## Prerequisites

- UNIX based system
- GNU Make
- Python 3.7

## Setup instructions

Use `make help` to display help information about available commands.

### Viewing results

1. Create and activate virtual environment:
    ```bash
    make create_environment; source source rs-venv/bin/activate
    ```
2. Install packages:
    ```bash
    make app_requirements
    ```
3. Run the app:
    ```bash
    make app
    ```

### Reproducing the analysis

1. Create and activate virtual environment:
    ```bash
    make create_environment
    source source rs-venv/bin/activate
    ```
2. Install packages required for development:
    ```bash
    make requirements
    ```
3. Run the whole pipeline:
    ```bash
    make scores
    ```
    - To check pipeline on small subset of data use `TEST_RUN=1` parameter.
    - To utilize make's parallel option using `-j <number_of_parallel_jobs>`, where `<number_of_parallel_jobs>` can be equal to number of processors cores.

## Project structure

    ├── LICENSE
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
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt        <- The requirements file for results presentation.
    ├── requirements-dev.txt    <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py           <- Project's main module. Can be installed with pip command.
    └── booksuggest                <- Source code for use in this project.
        │
        ├── data           <- Scripts to download or generate data.
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │                   predictions.
        │
        └── evaluation     <- Scripts to evaluate scores and validate results against ground-truth data.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>