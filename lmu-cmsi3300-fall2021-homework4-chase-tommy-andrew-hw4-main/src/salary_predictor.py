'''
salary_predictor.py
Predictor of salary from old census data -- riveting!
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

class SalaryPredictor:

    def __init__(self, X_train, y_train):
        """
        Creates a new SalaryPredictor trained on the given features from the
        preprocessed census data to predicted salary labels. Performs and fits
        any preprocessing methods (e.g., imputing of missing features,
        discretization of continuous variables, etc.) on the inputs, and saves
        these as attributes to later transform test inputs.

        :param DataFrame X_train: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_train: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        dfDesc = X_train.filter(['work_class','education','marital','occupation_code','relationship','race','sex','country'])
        dfCont = X_train.filter(['age','education_years','capital_gain','capital_loss','hours_per_week']).to_numpy()
        self.enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
        self.enc = self.enc.fit(dfDesc)
        dfDesc = self.enc.transform(dfDesc).toarray()
        listName = []
        for i in range(0,len(dfDesc)):
            listName.append(np.concatenate((dfDesc[i],dfCont[i])))
        self.logReg = LogisticRegression(class_weight = {'<=50K':1.4}).fit(listName,y_train)
        return


    def hotIt(self, data):
        dfDesc = data.filter(['work_class','education','marital','occupation_code','relationship','race','sex','country'])
        dfCont = data.filter(['age','education_years','capital_gain','capital_loss','hours_per_week']).to_numpy()
        enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
        dfDesc = self.enc.transform(dfDesc).toarray()
        listName = []
        for i in range(0,len(dfDesc)):
            listName.append(np.concatenate((dfDesc[i],dfCont[i])))
        return listName

    def classify (self, X_test):
        """
        Takes a DataFrame of rows of input attributes of census demographic
        and provides a classification for each. Note: must perform the same
        data transformations on these test rows as was done during training!

        :param DataFrame X_test: DataFrame of rows consisting of demographic
        attributes to be classified
        :return: A list of classifications, one for each input row X=x
        """
        y_pred = self.logReg.predict(self.hotIt(X_test))
        return y_pred

    def test_model (self, x_test, y_test):
        """
        Takes the test-set as input (2 DataFrames consisting of test demographics
        and their associated labels), classifies each, and then prints
        the classification_report on the expected vs. given labels.

        :param DataFrame X_test: Pandas DataFrame consisting of the
        sample rows of attributes pertaining to each individual
        :param DataFrame y_test: Pandas DataFrame consisting of the
        sample rows of labels pertaining to each person's salary
        """
        y_pred = self.classify(x_test)
        print(classification_report(y_pred, y_test))
        return classification_report(y_pred, y_test)


def load_and_sanitize (data_file):
    """
    Takes a path to the raw data file (a csv spreadsheet) and
    creates a new Pandas DataFrame from it with the sanitized
    data (e.g., removing leading / trailing spaces).
    NOTE: This should *not* do the preprocessing like turning continuous
    variables into discrete ones, or performing imputation -- those
    functions are handled in the SalaryPredictor constructor, and are
    used to preprocess all incoming test data as well.

    :param string data_file: String path to the data file csv to
    load-from and fashion a DataFrame from
    :return: The sanitized Pandas DataFrame containing the demographic
    information and labels. It is assumed that for n columns, the first
    n-1 are the inputs X and the nth column are the labels y
    """
    df = pd.read_csv(data_file)
    df = df.replace({' ':''},regex=True)
    return df


if __name__ == "__main__":
    data = load_and_sanitize('/Users/chasecour/Desktop/Sen-School/AI/lmu-cmsi3300-fall2021-homework4-chase-tommy-andrew-hw4-main/dat/salary.csv')
    x = data[['work_class','education','marital','occupation_code','relationship','race','sex','country','age','education_years','capital_gain','capital_loss','hours_per_week']]
    y_train, y_test, x_train, x_test = train_test_split(data['class'],x, test_size=0.11)
    s = SalaryPredictor(x_train, y_train)
    s.test_model(x_test, y_test)
    print(s.test_model(x_test, y_test))

    """
    df['work_class'] = df['work_class'].replace(' ?','work_no_class')
    dSetProb = df['work_class'].value_counts(normalize = True)
    np.random.choice(dSetProb,1,p=dSetProb)
    print(df.head(10))
    print("********")
    print(dSetProb)
    for i in df['work_class']:
        if i == ' ?':
            i = df['work_class'].sample(n=1,random_state=1)
    df['work_class'] = df['work_class'].replace(' ?','work_no_class')
    print(df['work_class'].sample(n=5,random_state=1))
    """
