
.PHONY: clean data lint requirements app tests docs

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = recommendation-system
VENV_NAME = rs-venv
PYTHON_INTERPRETER = python3.7
TEST_RUN=0

RAW_DATA_FILES = data/raw/book_tags.csv data/raw/book.csv data/raw/ratings.csv data/raw/tags.csv data/raw/to_read.csv data/raw/books_xml.zip

include cb-pipeline.mk cf-pipeline.mk

# Unified parts of the pipeline
MODELS = $(CB_MODELS) $(CF_MODELS)
APP_CB_MODELS = $(CB_MODELS)
APP_CF_MODELS = $(CF_MODELS)
PREDICTIONS = $(CB_PREDICTIONS)
SCORES = $(CB_SCORES) $(CF_SCORES)

#################################################################################
# COMMANDS                                                                      #
################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) setup.py install
	pip install numpy==1.15.4 # due to scikit-surprise installation dependency issue: https://github.com/NicolasHug/Surprise/issues/187
	pip install -r requirements.txt

## Download Dataset
data: $(RAW_DATA_FILES)

## Build features
features: $(TAG_FEATURES)

## Train models
models: $(MODELS)

## Run all tests
tests: 
	pytest

## Predict models
predictions: $(PREDICTIONS)

## Evaluate models
scores: $(SCORES) 

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .mypy_cache

## Delete all downloaded and calculated files
hard_clean: clean
	rm -rf data/raw/books_xml
	find data/raw data/interim data/processed ! -name '.gitkeep' -type f -delete
	find features -type f -name '*.csv' -delete
	find models -type f -name '*.pkl' -delete
	find models -type f -name '*.csv' -delete
	find results -type f -name '*.csv' -delete
	find app/assets/models -type f -name '*.pkl' -delete

## Lint using flake8 and check types with mypy
lint:
	flake8 src
	pylint src
	mypy src --ignore-missing-imports

## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv ${VENV_NAME}

## Start web application
app: models
	cp --update $(APP_CB_MODELS) app/assets/models/cb
	cp --update $(APP_CF_MODELS) app/assets/models/cf
	$(PYTHON_INTERPRETER) app/app.py

## Generate documentation
docs: 
	$(PYTHON_INTERPRETER) setup.py install
	make -C docs/ html

################################################################################
#
# Dataset cleaning rules 
#
################################################################################

BOOKS_XML_DIR = data/raw/books_xml

$(BOOKS_XML_DIR): data/raw/books_xml.zip
	$(PYTHON_INTERPRETER) -m src.data.extract_xml_files data/raw/books_xml.zip data/raw

data/processed/book.csv: $(RAW_DATA_FILES) $(BOOKS_XML_DIR)
	$(PYTHON_INTERPRETER) -m src.data.clean_book data/raw/book.csv $(BOOKS_XML_DIR) $@

data/processed/similar_books.csv: $(BOOKS_XML_DIR) data/processed/book.csv
	$(PYTHON_INTERPRETER) -m src.data.prepare_similar_books $(BOOKS_XML_DIR) data/processed/book.csv $@

data/processed/book_tags.csv: $(RAW_DATA_FILES)
	$(PYTHON_INTERPRETER) -m  src.data.clean_book_tags data/processed/book.csv data/raw/book_tags.csv data/raw/tags.csv data/external/genres.txt data/processed/book_tags.csv

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
	$(PYTHON_INTERPRETER) -m src.data.download_dataset $(book_tags_url) $@

data/raw/book.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) -m src.data.download_dataset $(books_url) $@
ifeq ($(TEST_RUN), 1)
	$(PYTHON_INTERPRETER) -m src.data.minify_dataframe $@ --n 100
endif

data/raw/ratings.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) -m src.data.download_dataset $(ratings_url) $@
ifeq ($(TEST_RUN), 1)
	$(PYTHON_INTERPRETER) -m src.data.minify_dataframe $@ --n 1000
endif

data/raw/tags.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) -m src.data.download_dataset $(tags_url) $@

data/raw/to_read.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) -m src.data.download_dataset $(to_read_url) $@
ifeq ($(TEST_RUN), 1)
	$(PYTHON_INTERPRETER) -m src.data.minify_dataframe $@ --n 1000
endif

data/raw/books_xml.zip: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) -m src.data.download_dataset $(books_xml_zip) $@

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
