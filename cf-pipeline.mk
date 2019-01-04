#################################################################################
# Collaborative filtering pipeline                                              #
################################################################################

# MODELS
CF_MODELS_DIR = models/collaborative-filtering-models

SLOPEONE_MODEL = $(CF_MODELS_DIR)/slopeone-model.pkl
KNN_MODEL = $(CF_MODELS_DIR)/knn-model.pkl
SVD_MODEL = $(CF_MODELS_DIR)/svd-model.pkl

CF_MODELS = $(SLOPEONE_MODEL) $(KNN_MODEL) $(SVD_MODEL)

# PREDICTIONS
CF_PREDICTIONS_DIR = models/predictions/cf-results
SLOPEONE_PREDICTION = $(CF_PREDICTIONS_DIR)/slopeone-predictions.csv
KNN_PREDICTION = $(CF_PREDICTIONS_DIR)/knn-predictions.csv
SVD_PREDICTION = $(CF_PREDICTIONS_DIR)/svd-predictions.csv

CF_PREDICTIONS = $(SLOPEONE_PREDICTION) $(KNN_PREDICTION) $(SVD_PREDICTION)

CF_ACCURACY_SCORES = results/cf-accuracy-results.csv
CF_EFFECTIVENESS_SCORES = results/cf-effectiveness-results.csv
CF_SCORES = $(CF_ACCURACY_SCORES) $(CF_EFFECTIVENESS_SCORES)

################################################################################
#
# Data preparation rules
#
################################################################################

data/processed/ratings-train.csv: data/raw/ratings.csv
	$(PYTHON_INTERPRETER) -m src.data.ratings_train_test_split $< data/processed/ratings-train.csv data/processed/ratings-test.csv

data/processed/ratings-test.csv: data/processed/ratings-train.csv

data/processed/to_read.csv: data/raw/to_read.csv data/processed/ratings-train.csv
	$(PYTHON_INTERPRETER) -m src.data.clean_to_read $< data/processed/ratings-train.csv $@

################################################################################
#
# Model training rules
#
################################################################################


$(SLOPEONE_MODEL): data/processed/ratings-train.csv
	$(PYTHON_INTERPRETER) -m src.models.cf_slopeone_models $< $@

$(KNN_MODEL): data/processed/ratings-train.csv
	$(PYTHON_INTERPRETER) -m src.models.cf_knn_models $< $@

$(SVD_MODEL): data/processed/ratings-train.csv
	$(PYTHON_INTERPRETER) -m src.models.cf_svd_models $< $@

################################################################################
#
# Model predictions rules
#
################################################################################

TASKS = task0 task1 task2 task3

$(SLOPEONE_PREDICTION): MODEL := $(SLOPEONE_MODEL)
$(SLOPEONE_PREDICTION): PREDICTION := $(SLOPEONE_PREDICTION)
$(SLOPEONE_PREDICTION): $(SLOPEONE_MODEL) $(TASKS:task%=cf_pred_task_so_%)

$(KNN_PREDICTION): MODEL := $(KNN_MODEL)
$(KNN_PREDICTION): PREDICTION := $(KNN_PREDICTION)
$(KNN_PREDICTION): $(KNN_MODEL) $(TASKS:task%=cf_pred_task_knn_%)

$(SVD_PREDICTION): MODEL := $(SVD_MODEL)
$(SVD_PREDICTION): PREDICTION := $(SVD_PREDICTION)
$(SVD_PREDICTION): $(SVD_MODEL) $(TASKS:task%=cf_pred_task_svd_%)

cf_pred_task%: $(CF_MODELS)
	$(PYTHON_INTERPRETER) -m src.models.cf_predict_models $(MODEL) $(PREDICTION) --n 10 --chunk-count 4 --chunk $(lastword $(subst _, ,$*))

################################################################################
#
# Model evaluation rules
#
################################################################################

$(CF_ACCURACY_SCORES): data/processed/ratings-test.csv  src/validation/cf_accuracy_evaluation.py $(CF_MODELS)
	$(PYTHON_INTERPRETER) -m src.validation.cf_accuracy_evaluation $(CF_MODELS_DIR) $< $@

$(CF_EFFECTIVENESS_SCORES): data/processed/to_read.csv data/processed/ratings-test.csv src/validation/cf_effectiveness_evaluation.py  $(CF_PREDICTIONS)
	$(PYTHON_INTERPRETER) -m src.validation.cf_effectiveness_evaluation $(CF_PREDICTIONS_DIR) data/processed/to_read.csv data/processed/ratings-test.csv $@
