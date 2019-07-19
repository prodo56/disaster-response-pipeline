from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
#download required nltk packages
nltk.download('stopwords')
# nltk.download('wordnet')

#load stop_words from nltk corpus
stop_words = stopwords.words("english")


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            if len(self.columns)>1:
                return X[self.columns].values
            else:
                return X[self.columns[0]].values
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
            
class Onehotencoder(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return pd.get_dummies(X).values


def tokenize(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens