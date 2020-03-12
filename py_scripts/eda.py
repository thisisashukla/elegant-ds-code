import os
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


class Explore():

    def __init__(self, df):
        self.df = df

    def describe(self):
        descr = self.df.describe().T
        na_counts = self.completeness()
        descr.loc[:, 'na %'] = na_counts/descr['count']
        
        self.descr = descr.T

    def completeness(self):
        return self.df.isna().sum(axis = 0)

    
    