import os
import time
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 


class Explore():

    def __init__(self, df, num_cols = None, cat_cols = None):
        self.df = df
        self.num_descr = None
        self.cat_descr = None
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def numDescribe(self):
        if isinstance(self.num_cols, set):
            descr = self.df[self.num_cols].describe().T
        else:
            descr = self.df.describe().T
            self.num_cols = set(descr.index.tolist())
        descr.loc[:, 'na%'] = (self.naCount(descr.index)/descr['count'])*100
        descr.loc[:, 'unique'] = [len(self.df[x].unique()) for x in descr.index]
        descr.loc[descr['count'] == 0, 'na%'] = np.nan

        cols = ['count', 'na%', 'unique', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        self.num_descr = descr[cols].T
        # self.num_cols = self.num_descr.columns.tolist()

        return self.num_descr

    def catDescribe(self):
        try:
            assert isinstance(self.num_descr, pd.DataFrame)
        except:
            print('Warning: numDescribe function needs to be called before catDescribe')
            self.numDescribe()
        if isinstance(self.cat_cols, set):
            pass
        else:
            self.cat_cols = set(self.df.columns)
            self.cat_cols.difference_update(self.num_cols)

        descr = pd.DataFrame(columns = ['count', 'na%', 'unique'], index = self.cat_cols)
        descr.loc[:, 'count'] = self.df[self.cat_cols].count()
        descr.loc[:, 'na%'] = (self.naCount(descr.index)/descr['count'])*100
        descr.loc[:, 'unique'] = [len(self.df[x].unique()) for x in descr.index]
        self.cat_descr = descr.T

        return self.cat_descr

    def naCount(self, cols):
        return self.df[cols].isna().sum(axis = 0)    
    
    def univariatePlots(self, c = 5):

        r = int(np.ceil(len(self.num_cols)/c))
        f, ax = plt.subplots(r, c, figsize = (5*r, 5*c))
        for i, col in enumerate(self.num_cols):
            k = i%c
            j = int((i-k)/c)
            pl = sns.distplot(self.df[self.df[col].isna() == False][col], ax = ax[j][k])
            pl.set_title('{} Distribution'.format(col))

        plt.show()

    def modifyCols(self, num_cols = set(), cat_cols = set()):

        if isinstance(num_cols, list):
            num_cols = set(num_cols)
        if isinstance(cat_cols, list):
            cat_cols = set(cat_cols)

        assert not bool(num_cols.intersection(cat_cols))

        self.num_cols.update(num_cols)
        self.num_cols.difference_update(cat_cols)
        self.cat_cols.update(cat_cols)
        self.cat_cols.difference_update(num_cols)

        assert not bool(self.num_cols.intersection(self.cat_cols))
        assert len(self.num_cols) + len(self.cat_cols) == self.df.shape[1]
       
        _ = self.numDescribe()
        _ = self.catDescribe()



    