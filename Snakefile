import os

CONFIG_NAMES = [os.path.splitext(os.path.basename(config))[0] for config in os.listdir('configs')]
PYTHON_INTERPRETER = "python3.7"

print(expand("{config}", config = CONFIG_NAMES))

CB_TEST_CASES = "data/processed/similar_books.csv"
CB_RESULTS_DIR = "models/predictions/cb-results"
CB_SCORES = "results/cb-results.csv"

CLEAN_BOOKS_WITH_NOUNS = "data/interim/cb-tf-idf/book_with_nouns.csv"
SIMILAR_BOOKS = "data/processed/similar_books.csv"
BOOKS_XML_DIR = "data/raw/books_xml"

MODEL_REC_COUNT = 50
TEST_REC_COUNT = 20

rule all:
    input:
        CB_SCORES

rule embedding_tsv:
    input:
        expand("embeddings-tsv/{config_name}.tsv", config_name=CONFIG_NAMES)


rule model_to_tsv:
    input:
        "models/content-based-models/{config_name}_model.pkl"
    output:
        "embeddings-tsv/{config_name}.tsv"
    shell:
        "python -m booksuggest.features.extract_feature_vectors {input} {output}"


rule clean_description_with_nouns:
    input:
        "data/processed/book.csv"
    output:
        CLEAN_BOOKS_WITH_NOUNS
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.data.prepare_description {input} {output}"

rule models:
    input:
        config="configs/{config_name}.ini",
        books=CLEAN_BOOKS_WITH_NOUNS
    output:
        "models/content-based-models/{config_name}_model.pkl"
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.models.train_cb_models_embedding {input.books} {output} --config_filepath {input.config} --rec_count {TEST_REC_COUNT}"

rule scores:
    input:
        expand("models/predictions/cb-results/{config_name}_predictions.csv", config_name=CONFIG_NAMES)
    output:
        CB_SCORES
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.evaluation.cb_evaluation {CB_RESULTS_DIR} {SIMILAR_BOOKS} {output} --rec_count {TEST_REC_COUNT}"

rule predictions:
    input:
        model="models/content-based-models/{config_name}_model.pkl",
        test_cases=CB_TEST_CASES
    output:
        "{CB_RESULTS_DIR}/{config_name}_predictions.csv"
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.evaluation.cb_predict_models {input.model} {input.test_cases} {output} --rec_count {MODEL_REC_COUNT}"

#######################################################
# DATASET PREPARATION RULES
#######################################################

rule books_processed:
    input:
        raw_books="data/raw/book.csv",
        books_xml_dir=BOOKS_XML_DIR
    output:
        "data/processed/book.csv"
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.data.clean_book {input.raw_books} {input.books_xml_dir} {output}"

rule similar_books_processed:
    input:
        BOOKS_XML_DIR,
        "data/processed/book.csv"
    output:
        "data/processed/similar_books.csv"
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.data.prepare_similar_books {BOOKS_XML_DIR} data/processed/book.csv {output}"

rule extract_books_xml:
    input:
        "data/raw/books_xml.zip"
    output:
        directory(BOOKS_XML_DIR)
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.data.extract_xml_files {input} data/raw"

#######################################################
# DATASET DOWNLOAD RULES 
#######################################################
BOOK_TAGS_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv"
BOOKS_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
RATINGS_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
TAGS_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv"
TO_READ_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/to_read.csv"
BOOKS_XML_ZIP = "https://github.com/zygmuntz/goodbooks-10k/raw/master/books_xml/books_xml.zip"

rule download_book_tags:
    output:
        "data/raw/book_tags.csv"
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.data.download_dataset {BOOK_TAGS_URL} {output}"


rule download_book:
    output:
        "data/raw/book.csv"
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.data.download_dataset {BOOKS_URL} {output}"

rule download_ratings:
    output:
        "data/raw/ratings.csv"
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.data.download_dataset {RATINGS_URL} {output}"

rule download_tags:
    output:
        "data/raw/tags.csv"
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.data.download_dataset {TAGS_URL} {output}"

rule download_to_read:
    output:
        "data/raw/to_read.csv"
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.data.download_dataset {TO_READ_URL} {output}"

rule books_xml:
    output:
        "data/raw/books_xml.zip"
    shell:
        "{PYTHON_INTERPRETER} -m booksuggest.data.download_dataset {BOOKS_XML_ZIP} {output}"