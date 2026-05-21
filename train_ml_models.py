import pandas as pd
import time
from joblib import dump
import features as f
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

train_dataset_file = "train_dataset.csv"
train_dataset = pd.read_csv(train_dataset_file)
train_urls = train_dataset["URL"].values.tolist()
train_labels = [1 if label == "phishing" else 0 for label in train_dataset["Label"].values.tolist()]

lexical_feature_pipeline = f.get_lexical_features(minmax=False)
lexical_minmax_feature_pipeline = f.get_lexical_features(minmax=True)

tfidf_feature_pipeline = f.get_tfidf_features()

full_feature_pipeline = f.get_full_features(minmax=False)
full_minmax_feature_pipeline = f.get_full_features(minmax=True)

logistic_regression_classsifier = LogisticRegression(max_iter=1000)
linear_svm_classifier = LinearSVC(max_iter=1000)
complement_naive_bayes_classifier = ComplementNB()
knn_classifier = KNeighborsClassifier(n_neighbors=5)
decision_tree_classifier = DecisionTreeClassifier(random_state=1)
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_classifier = XGBClassifier(tree_method="hist", max_depth=4, n_estimators=100, subsample=0.8, colsample_bytree=0.3)

classifiers = [logistic_regression_classsifier, linear_svm_classifier, complement_naive_bayes_classifier, knn_classifier, decision_tree_classifier, random_forest_classifier, xgb_classifier]

for c in classifiers:
    if type(c).__name__ == "ComplementNB":
        features = [("lexical", lexical_minmax_feature_pipeline), ("tf-idf", tfidf_feature_pipeline), ("full", full_minmax_feature_pipeline)]
    else:
        features = [("lexical", lexical_feature_pipeline), ("tf-idf", tfidf_feature_pipeline), ("full", full_feature_pipeline)]
    for name, feats in features:
        model = Pipeline([
            ("features", feats),
            ("classifier", c)
        ])
        model.fit(train_urls, train_labels)
        dump(model, f"models/{name}/{type(c).__name__}.joblib")