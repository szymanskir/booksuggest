# %%
from sklearn.metrics import coverage_error, label_ranking_loss, label_ranking_average_precision_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

# %%
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
# Remove rare labels
books["vector"] = vectors.values.tolist()
# %%

# %%
# Encode labels
mlb = MultiLabelBinarizer()
tag_labels = books[["label1", "label2", "label3"]].values
mlb.fit(tag_labels)
labels = mlb.transform(tag_labels)
# %%
# Train, test split
X, y = pd.DataFrame(item for item in books["vector"]), labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)
# %%
dt_class = DecisionTreeClassifier()
dt_class.fit(X_train, y_train)
# %%
y_pred = dt_class.predict(X_test)
# %%
books["label1"].value_counts()

# %%
# Check against rando uniform guess
1 / len(books["label1"].unique())
# %%
print(coverage_error(y_test, y_pred))
print(label_ranking_loss(y_test, y_pred))
print(label_ranking_average_precision_score(y_test, y_pred))

# %%
