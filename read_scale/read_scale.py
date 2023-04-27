'''
This code snipet provides useful functions to load information from the dataset from
Gotzner et al. (2018).

The scale is slightly modified by
(i) changing several typos,
(ii) adding the pronouns used for each adj scale in human experiments,
(iii) changing discrete measures to continuous.
'''

import pandas as pd
import os


class NgScale():
    '''
    This is a class for the dataset from Gotzner et al. (2018).
    '''

    def __init__(self):
        '''
        Load the scalar diversity dataset used in Gotnzer et al. (2018).

            Returns:
                ng_scale (pd.DataFrame): a data frame of scalar diversity dataset.
        '''
        curr_dir = os.getcwd()
        parent_dir = os.path.dirname(curr_dir)
        scale_path = f'{parent_dir}/datasets/ngscale_cleaned.csv'
        try:
            ng_scale = pd.read_csv(scale_path)
        except Exception:
            raise ValueError('The scale dataset might be missing. Check your datasets folder.')

        self.scale = ng_scale

    def get_scalar_items(self):
        '''
        Load the scalar items used in Gotnzer et al. (2018).

            Returns:
                self.weak (list): a list of weak scalar adjetives
                self.strong (list): a list of strong scalar adjectives
        '''
        ws = self.scale['adjective pair']
        self.weak, self.strong = list(), list()
        for pair in ws:
            self.weak.append(pair.split('/')[0])
            self.strong.append(pair.split('/')[1])

        return self.weak, self.strong

    def get_pronouns(self):
        '''
        Load the pronouns used in Gotnzer et al. (2018).

            Returns:
                self.pronoun_dic (dict): a dict of scalar pairs with associated pronouns
        '''
        weak, strong = self.get_scalar_items()
        self.pronoun_dic = dict()
        for pronoun in self.scale['pronoun']:
            self.pronoun_dic[(weak, strong)] = pronoun

        return self.pronouns

    def get_factors_and_data(self):
        '''
        Load the data reported in Gotnzer et al. (2018).

            Returns:
                self.factors (dict): a dictionary of influential factors and data
        '''
        self.factors = dict()
        for factor in list(self.scale.columns)[1:-1]:
            self.factors[factor] = list(self.scale[factor])

        return self.factors
