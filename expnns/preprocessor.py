import copy
import numpy as np
import pandas as pd


def min_max_scale(df, continuous, min_vals=None, max_vals=None):
    df_copy = copy.copy(df)
    for i, name in enumerate(continuous):
        if min_vals is None:
            min_val = np.min(df_copy[name])
        else:
            min_val = min_vals[i]
        if max_vals is None:
            max_val = np.max(df_copy[name])
        else:
            max_val = max_vals[i]
        df_copy[name] = (df_copy[name] - min_val) / (max_val - min_val)
    return df_copy


class Preprocessor:
    """
    class for dataframe dataset preprocessing:
    ordinal features: value 0: [1, 0, 0, 0], value 1: [1, 1, 0, 0], value 2: [1, 1, 1, 0]
    discrete features: value 0: [1, 0, 0, 0], value 1: [0, 1, 0, 0], value 2: [0, 0, 1, 0]
    continuous features: (x - min) / (max - min)
    """

    def __init__(self, ordinal, discrete, columns):
        self.ordinal = ordinal
        self.discrete = discrete
        self.columns = columns
        self.enc_cols = columns
        self.feature_var_map = dict()

    def _encode_one_feature(self, df, name, feat_num, type):
        i = df.columns.get_loc(name)  # current feature index in the updated dataframe
        enc_idx = list(range(i, i + feat_num))
        df_front = df[self.enc_cols[:i]]
        df_back = df[self.enc_cols[i + 1:]]

        enc = df.values[:, i]
        enc = enc[~np.isnan(enc)]   # to avoid nan bugs from pd
        encoded = np.zeros((len(enc), feat_num))

        if type == "ordinal":
            encoded = self._encode_one_feature_ordinal(enc, encoded)
        elif type == "discrete":
            encoded = self._encode_one_feature_discrete(enc, encoded)

        cols = [name + "_" + str(j) for j in range(feat_num)]
        enc_df = pd.DataFrame(data=encoded, columns=cols)

        new_df = pd.concat([df_front, enc_df, df_back], axis=1)
        return new_df, enc_idx

    def _encode_one_feature_ordinal(self, enc, encoded):
        for loc, val in enumerate(enc):
            vals = val + 1
            encoded[int(loc), :int(vals)] = 1
        return encoded

    def _encode_one_feature_discrete(self, enc, encoded):
        for loc, val in enumerate(enc):
            encoded[int(loc), int(val)] = 1
        return encoded

    def encode_df(self, df):
        self.enc_cols = self.columns # reset encoded cols
        df_copy = copy.copy(df)
        for (i, name) in enumerate(self.columns):
            if name in self.ordinal:
                df_copy, self.feature_var_map[i] = self._encode_one_feature(df_copy, name, self.ordinal[name],
                                                                            "ordinal")
            elif name in self.discrete:
                df_copy, self.feature_var_map[i] = self._encode_one_feature(df_copy, name, self.discrete[name],
                                                                            "discrete")
            else:
                idx = df_copy.columns.get_loc(name)
                self.feature_var_map[i] = [idx]  # continuous
            self.enc_cols = list(df_copy.columns)
        return df_copy

    def encode_one(self, x):
        """
        encode one point
        :param x: numpy array, shaped (x,)
        :return: x_copy: numpy array
        """
        self.enc_cols = self.columns # reset encoded cols
        xpd = pd.DataFrame(data=x.reshape(1, -1), columns=self.columns)
        return self.encode_df(xpd)

    def inverse_df(self, df):
        df_copy = copy.copy(df)
        return df_copy

    def inverse_one(self, x):
        x_copy = copy.copy(x)
        return x_copy
