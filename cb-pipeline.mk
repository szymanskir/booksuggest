
#################################################################################
# Content based pipeline                                                        #
################################################################################

# DATA
CLEAN_DESCRIPTION_WITH_NOUNS = data/interim/cb-tf-idf/book_with_nouns.csv
CLEAN_DESCRIPTION_WITHOUT_NOUNS = data/interim/cb-tf-idf/book_without_nouns.csv

# FEATURES
TAG_FEATURES = features/tag_based_features.csv

CB_SCORES = results/cb-results.csv

# MODELS
CB_MODELS_DIR = models/content-based-models

## TF-IDF models
TF_IDF_NOUNS = $(CB_MODELS_DIR)/tf-idf-nouns-model.pkl
TF_IDF_NOUNS_2GRAMS = $(CB_MODELS_DIR)/tf-idf-nouns-2grams-model.pkl
TF_IDF_NOUNS_3GRAMS = $(CB_MODELS_DIR)/tf-idf-nouns-3grams-model.pkl
TF_IDF_NO_NOUNS = $(CB_MODELS_DIR)/tf-idf-no-nouns-model.pkl
TF_IDF_NO_NOUNS_2GRAMS = $(CB_MODELS_DIR)/tf-idf-no-nouns-2grams-model.pkl
TF_IDF_NO_NOUNS_3GRAMS = $(CB_MODELS_DIR)/tf-idf-no-nouns-3grams-model.pkl

## Count based models
COUNT_NOUNS = $(CB_MODELS_DIR)/count-nouns-model.pkl
COUNT_NO_NOUNS = $(CB_MODELS_DIR)/count-no-nouns-model.pkl
COUNT_NOUNS_2GRAMS = $(CB_MODELS_DIR)/count-nouns-2grams-model.pkl
COUNT_NOUNS_3GRAMS = $(CB_MODELS_DIR)/count-nouns-3grams-model.pkl
COUNT_NO_NOUNS_2GRAMS = $(CB_MODELS_DIR)/count-no-nouns-2grams-model.pkl
COUNT_NO_NOUNS_3GRAMS = $(CB_MODELS_DIR)/count-no-nouns-3grams-model.pkl

## TF-IDF with tags models
TF_IDF_NOUNS_TAGS = $(CB_MODELS_DIR)/tf-idf-nouns-tags-model.pkl
TF_IDF_NOUNS_2GRAMS_TAGS = $(CB_MODELS_DIR)/tf-idf-nouns-tags-2grams-model.pkl
TF_IDF_NOUNS_3GRAMS_TAGS = $(CB_MODELS_DIR)/tf-idf-nouns-tags-3grams-model.pkl
TF_IDF_NO_NOUNS_TAGS= $(CB_MODELS_DIR)/tf-idf-no-nouns-tags-model.pkl
TF_IDF_NO_NOUNS_2GRAMS_TAGS= $(CB_MODELS_DIR)/tf-idf-no-nouns-tags-2grams-model.pkl
TF_IDF_NO_NOUNS_3GRAMS_TAGS = $(CB_MODELS_DIR)/tf-idf-no-nouns-tags-3grams-model.pkl

## Count based with tags models
COUNT_NOUNS_TAGS = $(CB_MODELS_DIR)/count-nouns-tags-model.pkl
COUNT_NOUNS_2GRAMS_TAGS = $(CB_MODELS_DIR)/count-nouns-2grams-tags-model.pkl
COUNT_NOUNS_3GRAMS_TAGS = $(CB_MODELS_DIR)/count-nouns-3grams-tags-model.pkl
COUNT_NO_NOUNS_TAGS = $(CB_MODELS_DIR)/count-no-nouns-tags-model.pkl
COUNT_NO_NOUNS_2GRAMS_TAGS = $(CB_MODELS_DIR)/count-no-nouns-2grams-tags-model.pkl
COUNT_NO_NOUNS_3GRAMS_TAGS = $(CB_MODELS_DIR)/count-no-nouns-3grams-tags-model.pkl

## CB models groups
1GRAMS_MODELS_TAGS = $(TF_IDF_NOUNS_TAGS) \
		     $(TF_IDF_NO_NOUNS_TAGS) \
		     $(COUNT_NOUNS_TAGS) \
		     $(COUNT_NO_NOUNS_TAGS)

1GRAMS_MODELS = $(TF_IDF_NOUNS) \
		$(TF_IDF_NO_NOUNS) \
		$(COUNT_NOUNS) \
		$(COUNT_NO_NOUNS) \
		$(1GRAMS_MODELS_TAGS)

2GRAMS_MODELS_TAGS = $(TF_IDF_NOUNS_2GRAMS_TAGS) \
		     $(TF_IDF_NO_NOUNS_2GRAMS_TAGS) \
		     $(COUNT_NOUNS_2GRAMS_TAGS) \
		     $(COUNT_NO_NOUNS_2GRAMS_TAGS)

2GRAMS_MODELS = $(TF_IDF_NOUNS_2GRAMS) \
		$(TF_IDF_NO_NOUNS_2GRAMS) \
		$(COUNT_NOUNS_2GRAMS) \
		$(COUNT_NO_NOUNS_2GRAMS) \
		$(2GRAMS_MODELS_TAGS)

3GRAMS_MODELS_TAGS = $(TF_IDF_NOUNS_3GRAMS_TAGS) \
		     $(TF_IDF_NO_NOUNS_3GRAMS_TAGS) \
		     $(COUNT_NOUNS_3GRAMS_TAGS) \
		     $(COUNT_NO_NOUNS_3GRAMS_TAGS)

3GRAMS_MODELS = $(TF_IDF_NOUNS_3GRAMS) \
		$(TF_IDF_NO_NOUNS_3GRAMS) \
		$(COUNT_NOUNS_3GRAMS) \
		$(COUNT_NO_NOUNS_3GRAMS) \
		$(3GRAMS_MODELS_TAGS)

CB_MODELS = $(1GRAMS_MODELS) \
	    $(2GRAMS_MODELS) \
	    $(3GRAMS_MODELS)

# PREDICTIONS
CB_RESULTS_DIR = models/predictions/cb-results

## Tf-idf predictions
TF_IDF_NOUNS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-nouns-predictions.csv
TF_IDF_NO_NOUNS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-no-nouns-predictions.csv
TF_IDF_NOUNS_2GRAMS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-nouns-2grams-predictions.csv
TF_IDF_NO_NOUNS_2GRAMS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-no-nouns-2grams-predictions.csv
TF_IDF_NOUNS_3GRAMS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-nouns-3grams-predictions.csv
TF_IDF_NO_NOUNS_3GRAMS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-no-nouns-3grams-predictions.csv

TF_IDF_NOUNS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-nouns-tags-predictions.csv
TF_IDF_NO_NOUNS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-no-nouns-tags-predictions.csv
TF_IDF_NOUNS_2GRAMS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-nouns-2grams-tags-predictions.csv
TF_IDF_NO_NOUNS_2GRAMS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-no-nouns-2grams-tags-predictions.csv
TF_IDF_NOUNS_3GRAMS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-nouns-3grams-tags-predictions.csv
TF_IDF_NO_NOUNS_3GRAMS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/tf-idf-no-nouns-3grams-tags-predictions.csv

## Count based predictions
COUNT_NOUNS_PREDICTION = $(CB_RESULTS_DIR)/count-nouns-predictions.csv
COUNT_NO_NOUNS_PREDICTION = $(CB_RESULTS_DIR)/count-no-nouns-predictions.csv
COUNT_NOUNS_2GRAMS_PREDICTION = $(CB_RESULTS_DIR)/count-nouns-2grams-predictions.csv
COUNT_NOUNS_3GRAMS_PREDICTION = $(CB_RESULTS_DIR)/count-nouns-3grams-predictions.csv
COUNT_NO_NOUNS_2GRAMS_PREDICTION = $(CB_RESULTS_DIR)/count-no-nouns-2grams-predictions.csv
COUNT_NO_NOUNS_3GRAMS_PREDICTION = $(CB_RESULTS_DIR)/count-no-nouns-3grams-predictions.csv

COUNT_NOUNS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/count-nouns-tags-predictions.csv
COUNT_NO_NOUNS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/count-no-nouns-tags-predictions.csv
COUNT_NOUNS_2GRAMS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/count-nouns-2grams-tags-predictions.csv
COUNT_NOUNS_3GRAMS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/count-nouns-3grams-tags-predictions.csv
COUNT_NO_NOUNS_2GRAMS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/count-no-nouns-2grams-tags-predictions.csv
COUNT_NO_NOUNS_3GRAMS_TAGS_PREDICTION = $(CB_RESULTS_DIR)/count-no-nouns-3grams-tags-predictions.csv

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
		$(COUNT_NO_NOUNS_3GRAMS_PREDICTION) \
		$(TF_IDF_NOUNS_TAGS_PREDICTION) \
		$(TF_IDF_NO_NOUNS_TAGS_PREDICTION) \
		$(TF_IDF_NOUNS_2GRAMS_TAGS_PREDICTION) \
		$(TF_IDF_NO_NOUNS_2GRAMS_TAGS_PREDICTION) \
		$(TF_IDF_NOUNS_3GRAMS_TAGS_PREDICTION) \
		$(TF_IDF_NO_NOUNS_3GRAMS_TAGS_PREDICTION) \
		$(COUNT_NOUNS_TAGS_PREDICTION) \
		$(COUNT_NO_NOUNS_TAGS_PREDICTION) \
		$(COUNT_NOUNS_2GRAMS_TAGS_PREDICTION) \
		$(COUNT_NOUNS_3GRAMS_TAGS_PREDICTION) \
		$(COUNT_NO_NOUNS_2GRAMS_TAGS_PREDICTION) \
		$(COUNT_NO_NOUNS_3GRAMS_TAGS_PREDICTION) 

################################################################################
#
# Data preparation rules
#
################################################################################

$(CLEAN_DESCRIPTION_WITH_NOUNS): data/processed/book.csv src/data/prepare_description.py 
	$(PYTHON_INTERPRETER) -m src.data.prepare_description $< $@

$(CLEAN_DESCRIPTION_WITHOUT_NOUNS): data/processed/book.csv src/data/prepare_description.py 
	$(PYTHON_INTERPRETER) -m src.data.prepare_description $< $@ --remove_nouns

################################################################################
#
# Feature building rules
#
################################################################################

$(TAG_FEATURES): data/processed/book_tags.csv src/features/build_tag_features.py
	$(PYTHON_INTERPRETER) -m src.features.build_tag_features $< $@

################################################################################
#
# Model training rules
#
################################################################################

models/cb_dummy_model.pkl: src/models/cb_dummy_model.py
	$(PYTHON_INTERPRETER) -m src.models.cb_dummy_model $@

COMMON_CB_DEPS = src/models/tf_idf_models.py

$(CB_MODELS): $(COMMON_CB_DEPS)

# ngram prerequisites
$(1GRAMS_MODELS): NGRAM := 1
$(2GRAMS_MODELS): NGRAM := 2
$(3GRAMS_MODELS): NGRAM := 3

# noun/no-noun prerequisites
NOUN_MODELS = $(TF_IDF_NOUNS) \
	      $(TF_IDF_NOUNS_2GRAMS) \
	      $(TF_IDF_NOUNS_3GRAMS) \
	      $(TF_IDF_NOUNS_TAGS) \
	      $(TF_IDF_NOUNS_2GRAMS_TAGS) \
	      $(TF_IDF_NOUNS_3GRAMS_TAGS) \
	      $(COUNT_NOUNS) \
	      $(COUNT_NOUNS_2GRAMS) \
	      $(COUNT_NOUNS_3GRAMS) \
	      $(COUNT_NOUNS_TAGS) \
	      $(COUNT_NOUNS_2GRAMS_TAGS) \
	      $(COUNT_NOUNS_3GRAMS_TAGS)

NO_NOUN_MODELS = $(TF_IDF_NO_NOUNS) \
	      $(TF_IDF_NO_NOUNS_2GRAMS) \
	      $(TF_IDF_NO_NOUNS_3GRAMS) \
	      $(TF_IDF_NO_NOUNS_TAGS) \
	      $(TF_IDF_NO_NOUNS_2GRAMS_TAGS) \
	      $(TF_IDF_NO_NOUNS_3GRAMS_TAGS) \
	      $(COUNT_NO_NOUNS) \
	      $(COUNT_NO_NOUNS_2GRAMS) \
	      $(COUNT_NO_NOUNS_3GRAMS) \
	      $(COUNT_NO_NOUNS_TAGS) \
	      $(COUNT_NO_NOUNS_2GRAMS_TAGS) \
	      $(COUNT_NO_NOUNS_3GRAMS_TAGS)

$(NOUN_MODELS): $(CLEAN_DESCRIPTION_WITH_NOUNS)
$(NO_NOUN_MODELS): $(CLEAN_DESCRIPTION_WITHOUT_NOUNS)

$(NOUN_MODELS): DESCR_FILE := $(CLEAN_DESCRIPTION_WITH_NOUNS)
$(NO_NOUN_MODELS): DESCR_FILE := $(CLEAN_DESCRIPTION_WITHOUT_NOUNS)

# counter/tf-idf prerequisites
TF_IDF_MODELS = $(TF_IDF_NOUNS) \
		$(TF_IDF_NOUNS_2GRAMS) \
	        $(TF_IDF_NOUNS_3GRAMS) \
	        $(TF_IDF_NOUNS_TAGS) \
	        $(TF_IDF_NOUNS_2GRAMS_TAGS) \
	        $(TF_IDF_NOUNS_3GRAMS_TAGS) \
		$(TF_IDF_NO_NOUNS) \
	        $(TF_IDF_NO_NOUNS_2GRAMS) \
	        $(TF_IDF_NO_NOUNS_3GRAMS) \
	        $(TF_IDF_NO_NOUNS_TAGS) \
	        $(TF_IDF_NO_NOUNS_2GRAMS_TAGS) \
	        $(TF_IDF_NO_NOUNS_3GRAMS_TAGS) \

COUNT_MODELS = $(COUNT_NOUNS) \
	       $(COUNT_NOUNS_2GRAMS) \
	       $(COUNT_NOUNS_3GRAMS) \
	       $(COUNT_NOUNS_TAGS) \
	       $(COUNT_NOUNS_2GRAMS_TAGS) \
	       $(COUNT_NOUNS_3GRAMS_TAGS) \
	       $(COUNT_NO_NOUNS) \
	       $(COUNT_NO_NOUNS_2GRAMS) \
	       $(COUNT_NO_NOUNS_3GRAMS) \
	       $(COUNT_NO_NOUNS_TAGS) \
	       $(COUNT_NO_NOUNS_2GRAMS_TAGS) \
	       $(COUNT_NO_NOUNS_3GRAMS_TAGS)

$(TF_IDF_MODELS): TEXT_MODEL_FLAG := --tf_idf
$(COUNT_MODELS): TEXT_MODEL_FLAG := --count

# tag prerequisites
TAG_BASED_MODELS = $(1GRAMS_MODELS_TAGS) \
		   $(2GRAMS_MODELS_TAGS) \
		   $(3GRAMS_MODELS_TAGS)

$(TAG_BASED_MODELS): TAG_OPTION := --tag_features $(TAG_FEATURES)
$(TAG_BASED_MODELS): $(TAG_FEATURES)

REC_COUNT = 20

$(CB_MODELS): $(COMMON_CB_DEPS)
	$(PYTHON_INTERPRETER) -m src.models.tf_idf_models $(DESCR_FILE) $@ --ngrams $(NGRAM) $(TEXT_MODEL_FLAG) --n $(REC_COUNT) $(TAG_OPTION)

################################################################################
#
# Model predictions rules
#
################################################################################
CB_TEST_CASES = data/processed/similar_books.csv

$(TF_IDF_NOUNS_PREDICTION): MODEL := $(TF_IDF_NOUNS)
$(TF_IDF_NOUNS_PREDICTION): $(TF_IDF_NOUNS)

$(TF_IDF_NO_NOUNS_PREDICTION): MODEL := $(TF_IDF_NO_NOUNS)
$(TF_IDF_NO_NOUNS_PREDICTION): $(TF_IDF_NO_NOUNS)

$(TF_IDF_NOUNS_2GRAMS_PREDICTION): MODEL := $(TF_IDF_NOUNS_2GRAMS)
$(TF_IDF_NOUNS_2GRAMS_PREDICTION): $(TF_IDF_NOUNS_2GRAMS)

$(TF_IDF_NO_NOUNS_2GRAMS_PREDICTION): MODEL := $(TF_IDF_NO_NOUNS_2GRAMS)
$(TF_IDF_NO_NOUNS_2GRAMS_PREDICTION): $(TF_IDF_NO_NOUNS_2GRAMS)

$(TF_IDF_NOUNS_3GRAMS_PREDICTION): MODEL := $(TF_IDF_NOUNS_3GRAMS)
$(TF_IDF_NOUNS_3GRAMS_PREDICTION): $(TF_IDF_NOUNS_3GRAMS)

$(TF_IDF_NO_NOUNS_3GRAMS_PREDICTION): MODEL := $(TF_IDF_NO_NOUNS_3GRAMS)
$(TF_IDF_NO_NOUNS_3GRAMS_PREDICTION): $(TF_IDF_NO_NOUNS_3GRAMS)

$(TF_IDF_NOUNS_TAGS_PREDICTION): MODEL := $(TF_IDF_NOUNS_TAGS)
$(TF_IDF_NOUNS_TAGS_PREDICTION): $(TF_IDF_NOUNS_TAGS)

$(TF_IDF_NO_NOUNS_TAGS_PREDICTION): MODEL := $(TF_IDF_NO_NOUNS_TAGS)
$(TF_IDF_NO_NOUNS_TAGS_PREDICTION): $(TF_IDF_NO_NOUNS_TAGS)

$(TF_IDF_NOUNS_2GRAMS_TAGS_PREDICTION): MODEL := $(TF_IDF_NOUNS_2GRAMS_TAGS)
$(TF_IDF_NOUNS_2GRAMS_TAGS_PREDICTION): $(TF_IDF_NOUNS_2GRAMS_TAGS)

$(TF_IDF_NO_NOUNS_2GRAMS_TAGS_PREDICTION): MODEL := $(TF_IDF_NO_NOUNS_2GRAMS_TAGS)
$(TF_IDF_NO_NOUNS_2GRAMS_TAGS_PREDICTION): $(TF_IDF_NO_NOUNS_2GRAMS_TAGS)

$(TF_IDF_NOUNS_3GRAMS_TAGS_PREDICTION): MODEL := $(TF_IDF_NOUNS_3GRAMS_TAGS)
$(TF_IDF_NOUNS_3GRAMS_TAGS_PREDICTION): $(TF_IDF_NOUNS_3GRAMS_TAGS)

$(TF_IDF_NO_NOUNS_3GRAMS_TAGS_PREDICTION): MODEL := $(TF_IDF_NO_NOUNS_3GRAMS_TAGS)
$(TF_IDF_NO_NOUNS_3GRAMS_TAGS_PREDICTION): $(TF_IDF_NO_NOUNS_3GRAMS_TAGS)

$(COUNT_NOUNS_TAGS_PREDICTION): MODEL := $(COUNT_NOUNS_TAGS)
$(COUNT_NOUNS_TAGS_PREDICTION): $(COUNT_NOUNS_TAGS)

$(COUNT_NO_NOUNS_TAGS_PREDICTION): MODEL := $(COUNT_NO_NOUNS_TAGS) 
$(COUNT_NO_NOUNS_TAGS_PREDICTION): $(COUNT_NO_NOUNS_TAGS) 

$(COUNT_NOUNS_2GRAMS_TAGS_PREDICTION): MODEL := $(COUNT_NOUNS_2GRAMS_TAGS)
$(COUNT_NOUNS_2GRAMS_TAGS_PREDICTION): $(COUNT_NOUNS_2GRAMS_TAGS)

$(COUNT_NOUNS_3GRAMS_TAGS_PREDICTION): MODEL := $(COUNT_NOUNS_3GRAMS_TAGS)
$(COUNT_NOUNS_3GRAMS_TAGS_PREDICTION): $(COUNT_NOUNS_3GRAMS_TAGS)

$(COUNT_NO_NOUNS_2GRAMS_TAGS_PREDICTION): MODEL := $(COUNT_NO_NOUNS_2GRAMS_TAGS)
$(COUNT_NO_NOUNS_2GRAMS_TAGS_PREDICTION): $(COUNT_NO_NOUNS_2GRAMS_TAGS)

$(COUNT_NO_NOUNS_3GRAMS_TAGS_PREDICTION): MODEL := $(COUNT_NO_NOUNS_3GRAMS_TAGS)
$(COUNT_NO_NOUNS_3GRAMS_TAGS_PREDICTION): $(COUNT_NO_NOUNS_3GRAMS_TAGS)

$(COUNT_NOUNS_PREDICTION): MODEL := $(COUNT_NOUNS)
$(COUNT_NOUNS_PREDICTION): $(COUNT_NOUNS)

$(COUNT_NO_NOUNS_PREDICTION): MODEL := $(COUNT_NO_NOUNS)
$(COUNT_NO_NOUNS_PREDICTION): $(COUNT_NO_NOUNS)

$(COUNT_NOUNS_2GRAMS_PREDICTION): MODEL := $(COUNT_NOUNS_2GRAMS)
$(COUNT_NOUNS_2GRAMS_PREDICTION): $(COUNT_NOUNS_2GRAMS)

$(COUNT_NO_NOUNS_2GRAMS_PREDICTION): MODEL := $(COUNT_NO_NOUNS_2GRAMS)
$(COUNT_NO_NOUNS_2GRAMS_PREDICTION): $(COUNT_NO_NOUNS_2GRAMS)

$(COUNT_NOUNS_3GRAMS_PREDICTION): MODEL := $(COUNT_NOUNS_3GRAMS)
$(COUNT_NOUNS_3GRAMS_PREDICTION): $(COUNT_NOUNS_3GRAMS)

$(COUNT_NO_NOUNS_3GRAMS_PREDICTION): MODEL := $(COUNT_NO_NOUNS_3GRAMS)
$(COUNT_NO_NOUNS_3GRAMS_PREDICTION): $(COUNT_NO_NOUNS_3GRAMS)

$(CB_PREDICTIONS): $(CB_TEST_CASES)
	$(PYTHON_INTERPRETER) -m src.models.predict_models $(MODEL) $(CB_TEST_CASES) $@

################################################################################
#
# Model evaluation rules
#
################################################################################

SIMILAR_BOOKS = data/processed/similar_books.csv

$(CB_SCORES): src/validation/cb_evaluation.py $(SIMILAR_BOOKS) $(CB_PREDICTIONS)
	$(PYTHON_INTERPRETER) -m src.validation.cb_evaluation $(CB_RESULTS_DIR) $(SIMILAR_BOOKS) $@