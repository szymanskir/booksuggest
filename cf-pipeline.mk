#################################################################################
# Collaborative filtering pipeline                                              #
################################################################################

# MODELS
CF_MODELS_DIR = models/collaborative-filtering-models

SVD_MODEL = $(CF_MODELS_DIR)/svd-model.pkl
DUMMY_CF_MODEL = $(CF_MODELS_DIR)/cf_dummy-model.pkl

CF_MODELS = $(DUMMY_CF_MODEL) $(SVD_MODEL)

# PREDICTIONS
CF_PREDICTIONS_DIR = models/predictions/cf-results
SVD_PREDICTION = $(CF_PREDICTIONS_DIR)/svd-predictions.csv

CF_PREDICTIONS = $(SVD_PREDICTION)

CF_TEST_SCORES = results/cf-test-results.csv
CF_TO_READ_SCORES = results/cf-to_read-results.csv
CF_SCORES = $(CF_TEST_SCORES) $(CF_TO_READ_SCORES)

################################################################################
#
# Data preparation rules
#
################################################################################

data/processed/ratings-train.csv data/processed/ratings-test.csv: data/raw/ratings.csv
	$(PYTHON_INTERPRETER) -m src.data.ratings_train_test_split $< data/processed/ratings-train.csv data/processed/ratings-test.csv

data/processed/to_read.csv: data/raw/to_read.csv data/processed/ratings-train.csv
	$(PYTHON_INTERPRETER) -m src.data.clean_to_read $< data/processed/ratings-train.csv $@

################################################################################
#
# Model training rules
#
################################################################################

$(DUMMY_CF_MODEL): src/models/cf_dummy_model.py
	$(PYTHON_INTERPRETER) -m src.models.cf_dummy_model $@

$(SVD_MODEL): src/models/cf_svd_models.py src/models/cf_recommend_models.py data/processed/ratings-train.csv data/processed/ratings-test.csv
	$(PYTHON_INTERPRETER) -m src.models.cf_svd_models data/processed/ratings-train.csv $@ --n 10 

################################################################################
#
# Model predictions rules
#
################################################################################

$(SVD_PREDICTION): MODEL := $(SVD_MODEL)
$(SVD_PREDICTION): $(SVD_MODEL)

$(CF_PREDICTIONS):
	$(PYTHON_INTERPRETER) -m src.models.cf_predict_models $(MODEL) $@

################################################################################
#
# Model evaluation rules
#
################################################################################

$(CF_TEST_SCORES): src/validation/cf_testset_evaluation.py data/processed/ratings-test.csv
	$(PYTHON_INTERPRETER) -m src.validation.cf_testset_evaluation $(CF_MODELS_DIR) data/processed/ratings-test.csv $@

$(CF_TO_READ_SCORES): src/validation/cf_to_read_evaluation.py data/processed/to_read.csv $(CF_PREDICTIONS)
	$(PYTHON_INTERPRETER) -m src.validation.cf_to_read_evaluation $(CF_PREDICTIONS_DIR) data/processed/to_read.csv $@