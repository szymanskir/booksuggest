#%%
import pandas as pd
import numpy as np

#%%
vectors = pd.read_csv(
    "./embeddings-tsv/word2vec_100featureSize_10windowSize_40iters_median_model.tsv",
    header=None,
    sep="\t",
)
books = pd.read_csv(
    "./embeddings-tsv/word2vec_100featureSize_10windowSize_40iters_median_model-labels.tsv",
    sep="\t",
)


# %%
## Remove rare labels
books["vector"] = vectors.values.tolist()
books = books.groupby("label1").filter(lambda x: len(x) > 20)
# %%
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#%%
## Encode labels
le = preprocessing.LabelEncoder()
le.fit(books[["label1"]])
labels = le.transform(books[["label1"]])
#%%
## Train, test split
X, y = pd.DataFrame(item for item in books["vector"]), labels

data_dmatrix = xgb.DMatrix(data=pd.DataFrame(X), label=y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44, stratify=y
)
#%%
xgb_class = xgb.XGBClassifier()
xgb_class.fit(X_train, y_train)
#%%
y_pred = xgb_class.predict(X_test)
#%%
books["label1"].value_counts()

#%%
## Check against rando uniform guess
1 / len(books["label1"].unique())
# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=books["label1"].unique()))
