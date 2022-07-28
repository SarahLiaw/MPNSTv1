import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
warnings.simplefilter(action='ignore', category=FutureWarning)


def read_data(pathname):
    data_path = pathname
    data = pd.read_csv(data_path)

    data = data.iloc[:, 1:-1]  # Removes Library_ID, so that data ranges from Diagnosis->TF->DELFI->ichor.
    return data


def get_X(data):
    """
    Removes diagnosis.
    :param data: data from read_data
    :return: All data, including feature names except for diagnosis.
    """
    return data.iloc[:, 1:]


def get_y(data):
    """
    :param data: data from read_data
    :return: diagnosis of all samples
    """
    return data.Diagnosis


def standard_scaling(x):
    """
    :param x: Any x (could be train/test)
    :return: Transformed/Standardized features of x.
    """
    return StandardScaler().fit_transform(x)
