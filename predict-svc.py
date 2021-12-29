import lightgbm as lgb
# mount Google Drive
from os.path import expanduser
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
#from thundersvm import SVC

# Load files into DataFrames
X_train = pd.read_csv("./data/X_train.csv")
X_submission = pd.read_csv("./data/X_test.csv")

# Split training set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
        X_train.drop(['Score'], axis=1),
        X_train['Score'],
        test_size=1/4.0,
        random_state=0
    )

# This is where you can do more feature selection
X_train_processed = X_train
X_test_processed = X_test
X_submission_processed = X_submission

from sklearn.base import BaseEstimator, TransformerMixin
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]
class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]

clf = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('colext', TextSelector(field='Text')),
             #('vect', CountVectorizer()),
            ('tfidf', TfidfVectorizer(max_df=0.5)),
            #('tfidf', TfidfTransformer()),
        ])),
        ('summary', Pipeline([
        ('colesum', TextSelector(field='Summary')),
             #('vect', CountVectorizer()),
            ('tfidf', TfidfVectorizer(max_df=0.5)),
            #('tfidf', TfidfTransformer()),
        ])),
        ('intellect', Pipeline([
            ('wordext', NumberSelector(field='intellectual')),
            ('wscaler', StandardScaler()),
        ])),
        ('polT', Pipeline([
            ('wordext', NumberSelector(field='polarity')),
            ('wscaler', StandardScaler()),
        ])),
        ('polS', Pipeline([
            ('wordext', NumberSelector(field='TextBlobS')),
            ('wscaler', StandardScaler()),
        ])),
        ('Helpn', Pipeline([
            ('wordext', NumberSelector(field='HelpfulnessNumerator')),
            ('wscaler', StandardScaler()),
        ])),
        ('Helpd', Pipeline([
            ('wordext', NumberSelector(field='HelpfulnessDenominator')),
            ('wscaler', StandardScaler()),
         ])),
        ('Help', Pipeline([
            ('wordext', NumberSelector(field='Helpfulness')),
            ('wscaler', StandardScaler()),
        ])),
#         ('Time', Pipeline([
#             ('wordext', NumberSelector(field='Time')),
#             #('wscaler', StandardScaler()),
#         ])),
        ('upper', Pipeline([
            ('wordext', NumberSelector(field='upper')),
            ('wscaler', StandardScaler()),
        ])),
        ('pos', Pipeline([
            ('wordext', NumberSelector(field='very_pos')),
            ('wscaler', StandardScaler()),
        ])),
        ('neg', Pipeline([
            ('wordext', NumberSelector(field='very_neg')),
            ('wscaler', StandardScaler()),
        ])),
#         ('stopword', Pipeline([
#             ('wordext', NumberSelector(field='stopwords')),
#             #('wscaler', StandardScaler()),
#         ])),
    ])),
   ('clf', SVC(C=100, kernel='rbf'))
    ])

X_train_processed['Text'] = X_train_processed['Text'].fillna('')
X_train_processed['Summary'] = X_train_processed['Summary'].fillna('')
X_train_processed['comb_text'] = X_train_processed['comb_text'].fillna('')
# Learn the model
model = clf.fit(X_train_processed, Y_train)


X_test_processed['Text'] = X_test_processed['Text'].fillna('')
X_test_processed['Summary'] = X_test_processed['Summary'].fillna('')
Y_test_predictions = model.predict(X_test_processed)


# Evaluate your model on the testing set
print("RMSE on testing set = ", mean_squared_error(Y_test, Y_test_predictions))

# Plot a confusion matrix
X_submission_processed['Text'] = X_submission_processed['Text'].fillna('')
X_submission_processed['Summary'] = X_submission_processed['Summary'].fillna('')
X_submission['Score'] = model.predict(X_submission_processed)
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission.csv", index=False)