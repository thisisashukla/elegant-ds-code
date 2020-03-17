import pytest
import pandas as pd
import sys
sys.path.append('../')

from py_scripts.eda import Explore

@pytest.mark.explore
def test_num_describe(iris_data):

    exp = Explore(iris_data)
    descr = iris_data.describe()
    num_descr = exp.numDescribe()

    exp_df(exp.num_descr)
    assert num_descr.shape[1] == descr.shape[1]

@pytest.mark.explore
def test_cat_describe(iris_data, capsys):

    # testing for warning when catDescribe() is called before numDescribe()
    exp1 = Explore(iris_data)
    cat_descr = exp1.catDescribe()
    
    captured = capsys.readouterr()
    assert captured.out == "Warning: numDescribe function needs to be called before catDescribe\n"
    
    exp2 = Explore(iris_data)
    num_descr = exp2.numDescribe()
    cat_descr = exp2.catDescribe()
    num_cols = exp2.num_cols
    cat_cols = exp2.cat_cols
    
    captured = capsys.readouterr()
    assert captured.out == ""
    assert set(num_cols + cat_cols) == set(iris_data.columns.tolist())
    assert iris_data.describe().shape[1] == num_descr.shape[1]
    assert iris_data.shape[1] - iris_data.describe().shape[1] == cat_descr.shape[1]
    assert all(cat_descr.loc['count', :] == [iris_data[iris_data[col].isna() == False].shape[0] for col in cat_cols])
    exp_df(cat_descr)
    
def exp_df(df):
    assert isinstance(df, pd.DataFrame)
    assert df.isnull().sum().sum() == 0
    assert all(df.loc['na%', :] <= 100)
    assert all(df.loc['unique', :]/df.loc['count', :] <= 1)