'''
spam_filter.py
Spam v. Ham Classifier trained and deployable upon short
phone text messages.
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *

class SpamFilter:

    def __init__(self, text_train, labels_train):
        """
        Creates a new text-message SpamFilter trained on the given text
        messages and their associated labels. Performs any necessary
        preprocessing before training the SpamFilter's Naive Bayes Classifier.
        As part of this process, trains and stores the CountVectorizer used
        in the feature extraction process.

        :param DataFrame text_train: Pandas DataFrame consisting of the
        sample rows of text messages
        :param DataFrame labels_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each text message
        """
        self.vectorizer = CountVectorizer(stop_words = {'english'})
        self.features = self.vectorizer.fit_transform(text_train)
        self.classifier = MultinomialNB()
        self.classifier.fit(self.features, labels_train)
        return

    def classify (self, text_test):
        """
        Takes as input a list of raw text-messages, uses the SpamFilter's
        vectorizer to convert these into the known bag of words, and then
        returns a list of classifications, one for each input text

        :param list/DataFrame text_test: A list of text-messages (strings) consisting
        of the messages the SpamFilter must classify as spam or ham
        :return: A list of classifications, one for each input text message
        where index in the output classes corresponds to index of the input text.
        """
        test_features = self.vectorizer.transform(text_test)
        classifications = list((self.classifier.predict(test_features)))
        return classifications

    def test_model (self, text_test, labels_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test texts
        and their associated labels), classifies each text, and then prints
        the classification_report on the expected vs. given labels.

        :param DataFrame text_test: Pandas DataFrame consisting of the
        test rows of text messages
        :param DataFrame labels_test: Pandas DataFrame consisting of the
        test rows of labels pertaining to each text message
        """
        y_actual = self.classify(text_test)
        return classification_report(labels_test, y_actual)


def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with only the message
    texts and labels as the remaining columns.

    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the texts
    and labels
    """
    data = pd.read_csv(data_file, encoding='latin-1')
    data = data.dropna(axis='columns')
    data = data.rename(columns={'v1':'class', 'v2':'text'})
    return data


if __name__ == "__main__":
    data = load_and_sanitize('/Users/chasecour/Desktop/Sen-School/AI/lmu-cmsi3300-fall2021-homework4-chase-tommy-andrew-hw4-main/dat/texts.csv')
    print(data.head())
    y_train, y_test, x_train, x_test = train_test_split(data['class'], data['text'], test_size=.30)
    s = SpamFilter(x_train, y_train)
    s.test_model(x_test, y_test)
    print(s.test_model(x_test, y_test))
