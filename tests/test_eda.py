import pytest
import sys
sys.path.append('../')

from py_scripts.eda import Explore

@pytest.mark.explore
def test_eda(iris_data):

    exp = Explore(iris_data)
    exp.describe()
    