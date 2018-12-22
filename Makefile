.PHONY: clean data lint requirements tests docs

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = recommendation-system
VENV_NAME = rs-venv
PYTHON_INTERPRETER = python3.7

RAW_DATA_FILES = data/raw/book_tags.csv data/raw/book.csv data/raw/ratings.csv data/raw/tags.csv data/raw/to_read.csv data/raw/books_xml.zip


# Content Based Pipeline
CLEAN_DESCRIPTION_WITH_NOUNS = data/interim/cb-tf-idf/book.csv
CB_SCORES = results/cb-results.csv

## TF-IDF pipeline
### Basic model
BASIC_TF_IDF_MODEL = models/content-based-models/basic-tf-idf-model.pkl

## CB predictions
CB_RESULTS_DIR = models/predictions/cb-results
BASIC_TF_IDF_PREDICTION = $(CB_RESULTS_DIR)/basic-tf-idf-predictions.csv



# Unified parts of the pipeline
RESULT_FILES = $(CB_SCORES)
MODELS = models/dummy_model.pkl $(BASIC_TF_IDF_MODEL)
PREDICTIONS = $(BASIC_TF_IDF_PREDICTION)
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) setup.py install
	pip install -r requirements.txt


## Download Dataset
data: $(RAW_DATA_FILES)

## Train models
models: $(MODELS)

## Run all tests
tests: 
	pytest

## Predict models
predictions: $(PREDICTIONS)

## Evaluate models
scores: $(RESULT_FILES) 

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .mypy_cache

## Lint using flake8 and check types with mypy
lint:
	flake8 src
	pylint src
	mypy src --ignore-missing-imports

## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv ${VENV_NAME}

## Test if python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

## Generate documentation
docs: 
	$(PYTHON_INTERPRETER) setup.py install
	make -C docs/ html

################################################################################
#
# Dataset cleaning rules 
#
################################################################################

clean_data: data/interim/book-unified_ids.csv data/interim/similar_books.csv

data/interim/book-unified_ids.csv data/interim/similar_books-unified_ids.csv: src/data/parse_xml_files.py $(RAW_DATA_FILES)
	$(PYTHON_INTERPRETER) src/data/parse_xml_files.py data/raw/books_xml.zip data/interim
	$(PYTHON_INTERPRETER) src/data/unify_ids.py data/raw data/interim data/interim


################################################################################
#
# Dataset downloading rules
#
################################################################################


# Provide urls for downloading data
book_tags_url = https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv
books_url = https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv
ratings_url = https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv
tags_url = https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv
to_read_url = https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/to_read.csv
books_xml_zip = https://github.com/zygmuntz/goodbooks-10k/raw/master/books_xml/books_xml.zip


data/raw/book_tags.csv: src/data/download_dataset.py 
	$(PYTHON_INTERPRETER) src/data/download_dataset.py $(book_tags_url) $@

data/raw/book.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) src/data/download_dataset.py $(books_url) $@

data/raw/ratings.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) src/data/download_dataset.py $(ratings_url) $@

data/raw/tags.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) src/data/download_dataset.py $(tags_url) $@

data/raw/to_read.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) src/data/download_dataset.py $(to_read_url) $@

data/raw/books_xml.zip: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) src/data/download_dataset.py $(books_xml_zip) $@

################################################################################
#
# Data preparation rules
#
################################################################################

$(CLEAN_DESCRIPTION_WITH_NOUNS): data/interim/book-unified_ids.csv src/data/prepare_description.py 
	$(PYTHON_INTERPRETER) -m src.data.prepare_description $< $@


################################################################################
#
# Model training rules
#
################################################################################


models/dummy_model.pkl: src/models/dummy_model.py
	$(PYTHON_INTERPRETER) -m src.models.dummy_model $@

# Content-Based Models
$(BASIC_TF_IDF_MODEL): $(CLEAN_DESCRIPTION_WITH_NOUNS) src/models/tf_idf_models.py src/models/recommendation_models.py 
	$(PYTHON_INTERPRETER) -m src.models.tf_idf_models $< $@ --n 10 

################################################################################
#
# Model predictions rules
#
################################################################################
CB_TEST_CASES = data/interim/similar_books-unified_ids.csv

$(BASIC_TF_IDF_PREDICTION): $(BASIC_TF_IDF_MODEL)
	$(PYTHON_INTERPRETER) -m src.models.predict_models $< $(CB_TEST_CASES) $@

################################################################################
#
# Model evaluation rules
#
################################################################################

SIMILAR_BOOKS = data/interim/similar_books-unified_ids.csv

$(CB_SCORES): src/validation/evaluation.py data/interim/similar_books-unified_ids.csv $(PREDICTIONS)
	$(PYTHON_INTERPRETER) -m src.validation.evaluation $(CB_RESULTS_DIR) $(SIMILAR_BOOKS) $@

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
