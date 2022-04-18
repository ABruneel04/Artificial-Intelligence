import pandas as pd
from pomegranate import *
import math
import itertools
import unittest
import numpy as np

if __name__ == '__main__':
    """
    PROBLEM 2.1
    Using the Pomegranate Interface, determine the answers to the
    queries specified in the instructions.

    ANSWER GOES BELOW:
    """
    x = pd.read_csv("/Users/chasecour/Desktop/Sen-School/AI/lmu-cmsi3300-fall2021-homework2-andrew-chase-tommy-hw2-main/dat/adbot-data.csv")
    col_names = x.columns.values
    find = np.where(col_names == 'S')
    findItm = find[0][0]
    R = {"S": {0: 0, 1: 1776, 2: 500}}
    print(list(R.keys())[0])
    evid = {'Ad1':0,'Ad2':0}
    print("***********************************")
    model = BayesianNetwork.from_samples(x, state_names = col_names ,algorithm = 'exact')
    print(model.predict_proba(evid).item(findItm).probability(0))
    evid = {'G':1,'Ad1':0,'Ad2':1}
    print("***********************************")
    model = BayesianNetwork.from_samples(x, state_names = col_names ,algorithm = 'exact')
    print(model.predict_proba(evid).item(findItm).probability(0))
    evid = {'T':1,'H':1, 'Ad1':1,'Ad2':0}
    print("***********************************")
    model = BayesianNetwork.from_samples(x, state_names = col_names ,algorithm = 'exact')
    print(model.predict_proba(evid).item(findItm).parameters[0])
    # TODO: 2.1
