import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

@pytest.fixture
def iris_data():

    iris = load_iris()
    data = iris['data']
    target = iris['target']

    df = pd.DataFrame(data = data, columns = iris['feature_names'])
    df['target'] = np.apply_along_axis(lambda x: iris['target_names'][x], 0, target)
    
    return df

@pytest.fixture
def model_data():

    ids = np.arange(10)
    names = ['A', None, 'C', 'D', 'E', None, None, 'C', 'A', None]
    marks = [10, 50, 30, 20, None, 50, 30, 20, 90, None]
    subjects = ['Sci', 'Math', 'Comp', 'Sci', 'Math', 'Hindi', 'Comp', 'Hindi', 'Math', 'Math']
    ages = [18, 17, 16, 17, 18, 19, 20, 16, 18, 17]

    df = pd.DataFrame({'ID': ids, 'Name': names, 'Subject': subjects, 'Marks': marks, 'Age': ages})

    return df
