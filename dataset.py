import enum


class DataType(enum.Enum):
    DISCRETE = 0
    ORDINAL = 1
    CONTINUOUS_REAL = 2


class Dataset:
    """
        Dataset class providing the following information of the dataset
        (ordinal/discrete features are encoded, continuous features are min-max scaled so feature ranges are [0, 1])

        num_features: int, number of features in the original dataset before preprocessing
        num_variables: int, number of variables in the dataset after preprocessing
        feature_types: dict of {(int) feature index, (DataType) type of data}
        feat_var_map: dict of {(int) feature index, [corresponding variable indices]}

    """
    def __init__(self, num_features, num_var, feature_types, feat_var_map):
        self.num_features = num_features
        self.num_variables = num_var
        self.feature_types = feature_types
        self.feat_var_map = feat_var_map
