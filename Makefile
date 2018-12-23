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


# Content Based Pipeline
CLEAN_DESCRIPTION_WITH_NOUNS = data/interim/cb-tf-idf/book_with_nouns.csv
CLEAN_DESCRIPTION_WITHOUT_NOUNS = data/interim/cb-tf-idf/book_without_nouns.csv
CB_SCORES = results/cb-results.csv

## CB models
CB_MODELS_DIR = models/content-based-models

### TF-IDF models
TF_IDF_WITH_NOUNS = $(CB_MODELS_DIR)/tf-idf-nouns-model.pkl
TF_IDF_WITHOUT_NOUNS = $(CB_MODELS_DIR)/tf-idf-no-nouns-model.pkl
TF_IDF_WITH_NOUNS_2GRAMS = $(CB_MODELS_DIR)/tf-idf-nouns-2grams-model.pkl
TF_IDF_WITH_NOUNS_3GRAMS = $(CB_MODELS_DIR)/tf-idf-nouns-3grams-model.pkl
TF_IDF_WITHOUT_NOUNS_2GRAMS = $(CB_MODELS_DIR)/tf-idf-no-nouns-2grams-model.pkl
TF_IDF_WITHOUT_NOUNS_3GRAMS = $(CB_MODELS_DIR)/tf-idf-no-nouns-3grams-model.pkl

### Count based models
COUNT_WITH_NOUNS = $(CB_MODELS_DIR)/count-nouns-model.pkl
COUNT_WITHOUT_NOUNS = $(CB_MODELS_DIR)/count-no-nouns-model.pkl
COUNT_WITH_NOUNS_2GRAMS = $(CB_MODELS_DIR)/count-nouns-2grams-model.pkl
COUNT_WITH_NOUNS_3GRAMS = $(CB_MODELS_DIR)/count-nouns-3grams-model.pkl
COUNT_WITHOUT_NOUNS_2GRAMS = $(CB_MODELS_DIR)/count-no-nouns-2grams-model.pkl
COUNT_WITHOUT_NOUNS_3GRAMS = $(CB_MODELS_DIR)/count-no-nouns-3grams-model.pkl

CB_MODELS = $(TF_IDF_WITH_NOUNS) \
	    $(TF_IDF_WITH_NOUNS_2GRAMS) \
	    $(TF_IDF_WITH_NOUNS_3GRAMS) \
	    $(TF_IDF_WITHOUT_NOUNS) \
	    $(TF_IDF_WITHOUT_NOUNS_2GRAMS) \
	    $(TF_IDF_WITHOUT_NOUNS_3GRAMS) \
	    $(COUNT_WITH_NOUNS) \
	    $(COUNT_WITHOUT_NOUNS) \
	    $(COUNT_WITH_NOUNS_2GRAMS) \
	    $(COUNT_WITH_NOUNS_3GRAMS) \
	    $(COUNT_WITHOUT_NOUNS_2GRAMS) \
	    $(COUNT_WITHOUT_NOUNS_3GRAMS)


## CB predictions
CB_RESULTS_DIR = models/predictions/cb-results

### Tf-idf predictions
TF_IDF_NOUNS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-nouns-predictions.csv
TF_IDF_NO_NOUNS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-no-nouns-predictions.csv
TF_IDF_NOUNS_2GRAMS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-nouns-2grams-predictions.csv
TF_IDF_NO_NOUNS_2GRAMS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-no-nouns-2grams-predictions.csv
TF_IDF_NOUNS_3GRAMS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-nouns-3grams-predictions.csv
TF_IDF_NO_NOUNS_3GRAMS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-no-nouns-3grams-predictions.csv

### Count based predictions
COUNT_NOUNS_PREDICTION = $(CB_RESULTS_DIR)/count-nouns-predictions.csv
COUNT_NO_NOUNS_PREDICTION = $(CB_RESULTS_DIR)/count-no-nouns-predictions.csv
COUNT_NOUNS_2GRAMS_PREDICTION = $(CB_RESULTS_DIR)/count-nouns-2grams-predictions.csv
COUNT_NOUNS_3GRAMS_PREDICTION = $(CB_RESULTS_DIR)/count-nouns-3grams-predictions.csv
COUNT_NO_NOUNS_2GRAMS_PREDICTION = $(CB_RESULTS_DIR)/count-no-nouns-2grams-predictions.csv
COUNT_NO_NOUNS_3GRAMS_PREDICTION = $(CB_RESULTS_DIR)/count-no-nouns-3grams-predictions.csv

CB_PREDICTIONS = $(TF_IDF_NOUNS_PREDICTION) \
		 $(TF_IDF_NO_NOUNS_PREDICTION) \
		 $(TF_IDF_NOUNS_2GRAMS_PREDICTION) \
		 $(TF_IDF_NO_NOUNS_2GRAMS_PREDICTION) \
		 $(TF_IDF_NOUNS_3GRAMS_PREDICTION) \
		 $(TF_IDF_NO_NOUNS_3GRAMS_PREDICTION) \
		 $(COUNT_NOUNS_PREDICTION) \
		 $(COUNT_NO_NOUNS_PREDICTION) \
		 $(COUNT_NOUNS_2GRAMS_PREDICTION) \
		 $(COUNT_NOUNS_3GRAMS_PREDICTION) \
		 $(COUNT_NO_NOUNS_2GRAMS_PREDICTION) \
		 $(COUNT_NO_NOUNS_3GRAMS_PREDICTION)

## SVD pipeline
### Basic model
BASIC_SVD_MODEL = models/collaborative-filtering-models/basic-svd-model.pkl

# Unified parts of the pipeline
RESULT_FILES = $(CB_SCORES)
MODELS = models/dummy_model.pkl $(CB_MODELS) $(BASIC_SVD_MODEL)
APP_MODELS = models/dummy_model.pkl $(CB_MODELS) $(BASIC_SVD_MODEL)
PREDICTIONS = $(CB_PREDICTIONS)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) setup.py install
	pip install numpy==1.15.4 # due to scikit-surprise installation dependency issue: https://github.com/NicolasHug/Surprise/issues/187
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

## Delete all downloaded and calculated files
hard_clean: clean
	find data/raw data/interim data/processed ! -name '.gitkeep' -type f -delete
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

# Start web application
app: models
	cp --update $(APP_MODELS) app/assets/models/
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
	$(PYTHON_INTERPRETER)-m  src.data.clean_book_tags data/processed/book.csv data/raw/book_tags.csv data/raw/tags.csv data/external/genres.txt data/processed/book_tags.csv

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
	$(PYTHON_INTERPRETER) -m src.data.minify_dataframe $@
endif

data/raw/ratings.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) -m src.data.download_dataset $(ratings_url) $@
ifeq ($(TEST_RUN), 1)
	$(PYTHON_INTERPRETER) -m src.data.minify_dataframe $@ --n 10000
endif

data/raw/tags.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) -m src.data.download_dataset $(tags_url) $@

data/raw/to_read.csv: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) -m src.data.download_dataset $(to_read_url) $@

data/raw/books_xml.zip: src/data/download_dataset.py
	$(PYTHON_INTERPRETER) -m src.data.download_dataset $(books_xml_zip) $@

################################################################################
#
# Data preparation rules
#
################################################################################

$(CLEAN_DESCRIPTION_WITH_NOUNS): data/processed/book.csv src/data/prepare_description.py 
	$(PYTHON_INTERPRETER) -m src.data.prepare_description $< $@

$(CLEAN_DESCRIPTION_WITHOUT_NOUNS): data/processed/book.csv src/data/prepare_description.py 
	$(PYTHON_INTERPRETER) -m src.data.prepare_description $< $@ --remove_nouns

data/processed/ratings-train.csv data/processed/ratings-test.csv: 
	$(PYTHON_INTERPRETER) -m src.data.ratings_train_test_split data/raw/ratings.csv data/processed/ratings-train.csv data/processed/ratings-test.csv

################################################################################
#
# Model training rules
#
################################################################################


models/dummy_model.pkl: src/models/dummy_model.py
	$(PYTHON_INTERPRETER) -m src.models.dummy_model $@

COMMON_CB_DEPS = src/models/tf_idf_models.py src/models/recommendation_models.py


TF_IDF_NOUNS_MODELS = $(TF_IDF_WITH_NOUNS) \
		      $(TF_IDF_WITH_NOUNS_2GRAMS) \
		      $(TF_IDF_WITH_NOUNS_3GRAMS)

TF_IDF_WITHOUT_NOUNS_MODELS = $(TF_IDF_WITHOUT_NOUNS) \
			      $(TF_IDF_WITHOUT_NOUNS_2GRAMS) \
			      $(TF_IDF_WITHOUT_NOUNS_3GRAMS)

TF_IDF_1GRAM_MODELS = $(TF_IDF_WITH_NOUNS) $(TF_IDF_WITHOUT_NOUNS)
TF_IDF_2GRAM_MODELS = $(TF_IDF_WITH_NOUNS_2GRAMS) $(TF_IDF_WITHOUT_NOUNS_2GRAMS)
TF_IDF_3GRAM_MODELS = $(TF_IDF_WITH_NOUNS_3GRAMS) $(TF_IDF_WITHOUT_NOUNS_3GRAMS)


COUNT_NOUNS_MODELS = $(COUNT_WITH_NOUNS) \
		     $(COUNT_WITH_NOUNS_2GRAMS) \
		     $(COUNT_WITH_NOUNS_3GRAMS)

COUNT_WITHOUT_NOUNS_MODELS = $(COUNT_WITHOUT_NOUNS) \
			     $(COUNT_WITHOUT_NOUNS_2GRAMS) \
			     $(COUNT_WITHOUT_NOUNS_3GRAMS)

TF_IDF_1GRAM_MODELS = $(TF_IDF_WITH_NOUNS) $(TF_IDF_WITHOUT_NOUNS)
TF_IDF_2GRAM_MODELS = $(TF_IDF_WITH_NOUNS_2GRAMS) $(TF_IDF_WITHOUT_NOUNS_2GRAMS)
TF_IDF_3GRAM_MODELS = $(TF_IDF_WITH_NOUNS_3GRAMS) $(TF_IDF_WITHOUT_NOUNS_3GRAMS)

COUNT_1GRAM_MODELS = $(COUNT_WITH_NOUNS) $(COUNT_WITHOUT_NOUNS)
COUNT_2GRAM_MODELS = $(COUNT_WITH_NOUNS_2GRAMS) $(COUNT_WITHOUT_NOUNS_2GRAMS)
COUNT_3GRAM_MODELS = $(COUNT_WITH_NOUNS_3GRAMS) $(COUNT_WITHOUT_NOUNS_3GRAMS)

$(TF_IDF_NOUNS_MODELS): $(CLEAN_DESCRIPTION_WITH_NOUNS) $(COMMON_CB_DEPS)
$(TF_IDF_WITHOUT_NOUNS_MODELS): $(CLEAN_DESCRIPTION_WITHOUT_NOUNS) $(COMMON_CB_DEPS)

$(COUNT_NOUNS_MODELS): $(CLEAN_DESCRIPTION_WITH_NOUNS) $(COMMON_CB_DEPS)
$(COUNT_WITHOUT_NOUNS_MODELS): $(CLEAN_DESCRIPTION_WITHOUT_NOUNS) $(COMMON_CB_DEPS)
REC_COUNT = 10

# Tf-idf models
$(TF_IDF_1GRAM_MODELS):
	$(PYTHON_INTERPRETER) -m src.models.tf_idf_models $< $@ --n $(REC_COUNT) 

$(TF_IDF_2GRAM_MODELS):
	$(PYTHON_INTERPRETER) -m src.models.tf_idf_models $< $@ --n $(REC_COUNT) --ngrams 2

$(TF_IDF_3GRAM_MODELS):
	$(PYTHON_INTERPRETER) -m src.models.tf_idf_models $< $@ --n $(REC_COUNT) --ngrams 3

# Count based models
$(COUNT_1GRAM_MODELS):
	$(PYTHON_INTERPRETER) -m src.models.tf_idf_models $< $@ --n $(REC_COUNT) --count

$(COUNT_2GRAM_MODELS):
	$(PYTHON_INTERPRETER) -m src.models.tf_idf_models $< $@ --n $(REC_COUNT) --ngrams 2 --count

$(COUNT_3GRAM_MODELS):
	$(PYTHON_INTERPRETER) -m src.models.tf_idf_models $< $@ --n $(REC_COUNT) --ngrams 3 --count

# Collaborative-Filtering Models
$(BASIC_SVD_MODEL): src/models/cf_svd_models.py src/models/recommendation_models.py data/processed/ratings-train.csv data/processed/ratings-test.csv
	$(PYTHON_INTERPRETER) -m src.models.cf_svd_models data/processed/ratings-train.csv $@ --n 10 

################################################################################
#
# Model predictions rules
#
################################################################################
CB_TEST_CASES = data/processed/similar_books.csv

$(TF_IDF_NOUNS_PREDICTION): $(TF_IDF_WITH_NOUNS)
$(TF_IDF_NO_NOUNS_PREDICTION): $(TF_IDF_WITHOUT_NOUNS)
$(TF_IDF_NOUNS_2GRAMS_PREDICTION): $(TF_IDF_WITH_NOUNS_2GRAMS)
$(TF_IDF_NO_NOUNS_2GRAMS_PREDICTION): $(TF_IDF_WITHOUT_NOUNS_2GRAMS)
$(TF_IDF_NOUNS_3GRAMS_PREDICTION): $(TF_IDF_WITH_NOUNS_3GRAMS)
$(TF_IDF_NO_NOUNS_3GRAMS_PREDICTION): $(TF_IDF_WITHOUT_NOUNS_3GRAMS)

$(COUNT_NOUNS_PREDICTION): $(COUNT_WITH_NOUNS)
$(COUNT_NO_NOUNS_PREDICTION): $(COUNT_WITHOUT_NOUNS)
$(COUNT_NOUNS_2GRAMS_PREDICTION): $(COUNT_WITH_NOUNS_2GRAMS)
$(COUNT_NO_NOUNS_2GRAMS_PREDICTION): $(COUNT_WITHOUT_NOUNS_2GRAMS)
$(COUNT_NOUNS_3GRAMS_PREDICTION): $(COUNT_WITH_NOUNS_3GRAMS)
$(COUNT_NO_NOUNS_3GRAMS_PREDICTION): $(COUNT_WITHOUT_NOUNS_3GRAMS)

$(CB_PREDICTIONS): 
	$(PYTHON_INTERPRETER) -m src.models.predict_models $< $(CB_TEST_CASES) $@

################################################################################
#
# Model evaluation rules
#
################################################################################

SIMILAR_BOOKS = data/processed/similar_books.csv

$(CB_SCORES): src/validation/evaluation.py data/processed/similar_books.csv $(PREDICTIONS)
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
