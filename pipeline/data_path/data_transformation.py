import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def read_data(data_path):
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


def standard_scaling(x_test):
    """
    :param x_test: For any training X
    :return: Transformed/Standardized features of x.
    """
    return StandardScaler().fit_transform(x_test)


def standard_scaling_pred(x_pred):
    """
    :param x_pred: For any testing X.
    :return: Transformed/Standardized features of x.
    """
    return StandardScaler().fit(x_pred)


def encode_target(y):
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(y)
    label_encoder_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")
    # print("Label Encoded Target Variable", encoded_y, sep="\n")
    return encoded_y
