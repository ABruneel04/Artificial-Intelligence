'''
ad_engine.py
Advertisement Selection Engine that employs a Decision Network
to Maximize Expected Utility associated with different Decision
variables in a stochastic reasoning environment.

@author: <Your Name(s) Here>
'''
import pandas as pd
from pomegranate import *
import math
import itertools
import unittest
import numpy as np
import copy


class AdEngine:

    def __init__(self, data_file, dec_vars, util_map):
        """
        Responsible for initializing the Decision Network of the
        AdEngine using the following inputs

        :param string data_file: path to csv file containing data on which
        the network's parameters are to be learned
        :param list dec_vars: list of string names of variables to be
        considered decision variables for the agent. Example:
          ["Ad1", "Ad2"]
        :param dict util_map: discrete, tabular, utility map whose keys
        are variables in network that are parents of a utility node, and
        values are dictionaries mapping that variable's values to a utility
        score, for example:
          {
            "X": {0: 20, 1: -10}
          }
        represents a utility node with single parent X whose value of 0
        has a utility score of 20, and value 1 has a utility score of -10
        """
        self.data_file = data_file
        self.dec_vars = dec_vars
        self.util_map = util_map
        self.x = pd.read_csv(self.data_file)
        self.col_names = self.x.columns.values
        self.utilityVar = list(self.util_map.keys())[0]
        self.colOfUtilityVar = self.x.columns.get_loc(self.utilityVar)
        self.model = BayesianNetwork.from_samples(self.x, state_names = self.col_names ,algorithm = 'exact')
        self.possibleDecisions = self.options()


    def options(self):
        """
        Explanation:
        This is a helper method that creates a tuple containing all the choices
        that can be made. It loops through the decision varibles and the keys
        to create all the combonations.
        """
        poss_vals = dict()
        for var in self.dec_vars:
            poss_vals[var] = self.x[var].unique()
        combinations = itertools.product(*(poss_vals[var] for var in poss_vals))
        comboOptions = tuple(combinations)
        return comboOptions

    def eu(self, evidence, decisionValue, decisionVarible):
        """
        Explanation:
        This is a helper method that is used to create the expected utility of
        a given decision. It takes the values of the varibles and what the
        varibles are named to pass them in as evidence and get the result. This
        was made to help make the meu able to call this for each situation.
        """
        totalUtility = 0
        for i in range(0,len(decisionValue)):
            evidence[decisionVarible[i]] = decisionValue[i]
        for varibleState in self.util_map[list(self.util_map.keys())[0]]:
            probList = self.model.predict_proba(evidence).item(self.colOfUtilityVar).probability(varibleState)
            utility = probList * self.util_map[self.utilityVar][varibleState]
            totalUtility = totalUtility + utility
        return totalUtility
    def meu(self, evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, selects the ad content that maximizes expected utility
        and returns a dictionary over any decision variables and their
        best values plus the MEU from making this selection.

        :param dict evidence: dict mapping network variables to their
        observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: 2-Tuple of the format (a*, MEU) where:
          - a* = dict of format: {"DecVar1": val1, "DecVar2": val2, ...}
          - MEU = float representing the EU(a* | evidence)

        Explanation:
        This method utalizes the eu method and the self.possibleDecisions to
        loop through all the decisions that you can make and their eu. It will
        replace the choice and the value if there is a new max value. It then
        returns the dictonary of choices and the value of the best decision.
        """

        maxDecEU = -math.inf
        currentDecisionValues = self.possibleDecisions[0]
        bestDecisionValues = currentDecisionValues
        for j in self.possibleDecisions:
            currentDecisionValues = j
            curDecEU = self.eu(evidence, currentDecisionValues, self.dec_vars)
            if curDecEU > maxDecEU:
                bestDecisionValues = currentDecisionValues
                maxDecEU = curDecEU
        dictOfBestDecisions = dict()
        for i in range (0,len(self.dec_vars)):
            dictOfBestDecisions[self.dec_vars[i]] = bestDecisionValues[i]
        best_decisions, best_util = dictOfBestDecisions, maxDecEU
        return (best_decisions, best_util)

    def vpi(self, potential_evidence, observed_evidence):
        """
        Given some observed demographic "evidence" about a potential
        consumer, returns the Value of Perfect Information (VPI)
        that would be received on the given "potential" evidence about
        that consumer.

        :param string potential_evidence: string representing the variable name
        of the variable under evaluation
        :param dict observed_evidence: dict mapping network variables
        to their observed values, of the format: {"Obs1": val1, "Obs2": val2, ...}
        :return: float value indicating the VPI(potential | observed)

        Explanation:
        This will loop through the possible options the potential_evidence and
        call meu. Then it multiplies the probibility that of the
        potential_evidence by the meu and sums over it. Finally returns the sum.
        """
        startingEvidence = dict()
        for i in observed_evidence:
            startingEvidence[i] = observed_evidence[i]
        obsEvdMEU = self.meu(observed_evidence)[1]
        potEvdVals = self.x[potential_evidence].unique()
        sum = 0
        colOfPotEvd = self.x.columns.get_loc(potential_evidence)
        for i in potEvdVals:
            observed_evidence[potential_evidence] = i
            potEvdMEU = self.meu(observed_evidence)[1]
            curValUtil = potEvdMEU * self.model.predict_proba(startingEvidence).item(colOfPotEvd).probability(i)
            sum = sum + curValUtil
        return sum - obsEvdMEU


class AdAgentTests(unittest.TestCase):

    def test_meu_lecture_example_no_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"D": 0}, decision[0])
        self.assertAlmostEqual(2, decision[1], delta=0.01)
    def test_meu_lecture_example_with_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {"X": 0}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"D": 1}, decision[0])
        self.assertAlmostEqual(2, decision[1], delta=0.01)

        evidence2 = {"X": 1}
        decision2 = ad_engine.meu(evidence2)
        self.assertEqual({"D": 0}, decision2[0])
        self.assertAlmostEqual(2.4, decision2[1], delta=0.01)

    def test_vpi_lecture_example_no_evidence(self):
        ad_engine = AdEngine('../dat/lecture5-2-data.csv', ["D"], {"Y": {0: 3, 1: 1}})
        evidence = {}
        vpi = ad_engine.vpi("X", evidence)
        self.assertAlmostEqual(0.24, vpi, delta=0.1)

    def test_meu_defendotron_no_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"Ad1": 1, "Ad2": 0}, decision[0])
        self.assertAlmostEqual(746.72, decision[1], delta=0.01)

    def test_meu_defendotron_with_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"T": 1}
        decision = ad_engine.meu(evidence)
        self.assertEqual({"Ad1": 1, "Ad2": 1}, decision[0])
        self.assertAlmostEqual(720.73, decision[1], delta=0.01)

        evidence2 = {"T": 0, "G": 0}
        decision2 = ad_engine.meu(evidence2)
        self.assertEqual({"Ad1": 0, "Ad2": 0}, decision2[0])
        self.assertAlmostEqual(796.82, decision2[1], delta=0.01)

    def test_vpi_defendotron_no_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(20.77, vpi, delta=0.1)

        vpi2 = ad_engine.vpi("F", evidence)
        self.assertAlmostEqual(0, vpi2, delta=0.1)

    def test_vpi_defendotron_with_evidence(self):
        ad_engine = AdEngine('../dat/adbot-data.csv', ["Ad1", "Ad2"], {"S": {0: 0, 1: 1776, 2: 500}})
        evidence = {"T": 0}
        vpi = ad_engine.vpi("G", evidence)
        self.assertAlmostEqual(25.49, vpi, delta=0.1)

        evidence2 = {"G": 1}
        vpi2 = ad_engine.vpi("P", evidence2)
        self.assertAlmostEqual(0, vpi2, delta=0.1)

        evidence3 = {"H": 0, "T": 1, "P": 0}
        vpi3 = ad_engine.vpi("G", evidence3)
        self.assertAlmostEqual(66.76, vpi3, delta=0.1)

if __name__ == '__main__':
    unittest.main()
