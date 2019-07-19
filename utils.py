from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

#download required nltk packages
nltk.download('stopwords')

#load stop_words from nltk corpus
stop_words = stopwords.words("english")


class ColumnSelector(BaseEstimator, TransformerMixin):
    '''
    ColumnSelector is a custom Estimator class to help select the given columns from the df required in
    the model
    '''
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
    '''
    Onehotencoder is a custom Estimator class to create columns for each target class and to set to 1/0
    '''
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return pd.get_dummies(X).values


def tokenize(text):
    '''
    tokenize the sentences and normalize the given input
    :param text: message text
    :return tokens: list of normalized tokens
    '''
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens