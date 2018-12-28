
#################################################################################
# Collaborative filtering pipeline                                              #
################################################################################

BASIC_SVD_MODEL = $(CF_MODELS_DIR)/basic-svd-model.pkl

DUMMY_CF_MODEL = $(CF_MODELS_DIR)/cf_dummy_model.pkl

CF_MODELS = $(DUMMY_CF_MODEL) $(BASIC_SVD_MODEL)
CF_SCORES = results/cf-results.csv
CF_MODELS_DIR = models/collaborative-filtering-models

################################################################################
#
# Data preparation rules
#
################################################################################

data/processed/ratings-train.csv data/processed/ratings-test.csv: data/raw/ratings.csv
	$(PYTHON_INTERPRETER) -m src.data.ratings_train_test_split $< data/processed/ratings-train.csv data/processed/ratings-test.csv

################################################################################
#
# Model training rules
#
################################################################################

$(DUMMY_CF_MODEL): src/models/cf_dummy_model.py
	$(PYTHON_INTERPRETER) -m src.models.cf_dummy_model $@

$(BASIC_SVD_MODEL): src/models/cf_svd_models.py src/models/cf_recommend_models.py data/processed/ratings-train.csv data/processed/ratings-test.csv
	$(PYTHON_INTERPRETER) -m src.models.cf_svd_models data/processed/ratings-train.csv $@ --n 10 

################################################################################
#
# Model predictions rules
#
################################################################################



################################################################################
#
# Model evaluation rules
#
################################################################################

$(CF_SCORES): data/processed/ratings-test.csv data/raw/to_read.csv $(CF_MODELS)
	$(PYTHON_INTERPRETER) -m src.validation.cf_evaluation $(CF_MODELS_DIR) data/processed/ratings-test.csv data/raw/to_read.csv $@