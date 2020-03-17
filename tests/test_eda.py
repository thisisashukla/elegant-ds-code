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
    num_cols.update(cat_cols)
    assert num_cols == set(iris_data.columns.tolist())
    assert iris_data.describe().shape[1] == num_descr.shape[1]
    assert iris_data.shape[1] - iris_data.describe().shape[1] == cat_descr.shape[1]
    assert all(cat_descr.loc['count', :] == [iris_data[iris_data[col].isna() == False].shape[0] for col in cat_cols])
    exp_df(cat_descr)
    
def test_modifyCols(model_data):

    exp = Explore(model_data)
    _ = exp.numDescribe()
    _ = exp.catDescribe()
    assert set(exp.num_cols) == set(['ID', 'Marks', 'Age'])
    assert set(exp.cat_cols) == set(['Name', 'Subject'])

    # exp.modifyCols()
    # assert set(exp.numDescribe().columns) == set(['ID', 'Marks', 'Age'])
    # assert set(exp.catDescribe().columns) == set(['Name', 'Subject'])

    exp.modifyCols(['Marks', 'Age'], ['ID', 'Name', 'Subject'])
    assert set(exp.numDescribe().columns) == set(['Marks', 'Age'])
    assert set(exp.catDescribe().columns) == set(['ID', 'Name', 'Subject'])

    exp.modifyCols(num_cols=set(['ID']))
    assert set(exp.numDescribe().columns) == set(['ID', 'Marks', 'Age'])
    assert set(exp.catDescribe().columns) == set(['Name', 'Subject'])

    with pytest.raises(Exception):
        exp.modifyCols(set('Marks'), set('Marks'))
    # assert set(exp.numDescribe().columns) == set(['ID', 'Marks', 'Age'])
    # assert set(exp.catDescribe().columns) == set(['Name', 'Subject'])

def exp_df(df):
    assert isinstance(df, pd.DataFrame)
    assert df.isnull().sum().sum() == 0
    assert all(df.loc['na%', :] <= 100)
    assert all(df.loc['unique', :]/df.loc['count', :] <= 1)