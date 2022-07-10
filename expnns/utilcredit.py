import pandas as pd
import numpy as np
from preprocessor import Preprocessor, min_max_scale

feature_encodings_old = {'checking-status': {'A14': int(0), 'A11': int(1), 'A12': int(2), 'A13': int(3)},
                         'credit-history': {'A30': int(2), 'A31': int(4), 'A32': int(3), 'A33': int(0), 'A34': int(1)},
                         'purpose': {'A410': int(0), 'A41': int(2), 'A42': int(3), 'A43': int(4), 'A44': int(5),
                                     'A45': int(6), 'A46': int(7), 'A47': int(8), 'A48': int(9), 'A49': int(10),
                                     'A40': int(1), },
                         'savings': {'A65': int(0), 'A61': int(1), 'A62': int(2), 'A63': int(3), 'A64': int(4), },
                         'employment': {'A71': int(0), 'A72': int(1), 'A73': int(2), 'A74': int(3), 'A75': int(4), },
                         'sex-status': {'A91': int(0), 'A92': int(1), 'A93': int(1), 'A94': int(2), 'A95': int(3)},
                         'guarantors': {'A101': int(0), 'A102': int(1), 'A103': int(2)},  # RESIDENCE NEEDS TO -1
                         'property': {'A121': int(3), 'A122': int(2), 'A123': int(1), 'A124': int(0)},
                         'installment': {'A141': int(0), 'A142': int(1), 'A143': int(2)},
                         'housing': {'A151': int(1), 'A152': int(2), 'A153': int(0)},  # NUM-CREDITS NEEDS TO -1
                         'job': {'A171': int(0), 'A172': int(1), 'A173': int(2), 'A174': int(3)},  # LIABLE NEEDS TO -1
                         'phone': {'A191': int(0), 'A192': int(1)},
                         'foreign': {'A201': int(0), 'A202': int(1)},
                         'good-credit': {1: int(1), 2: int(0)},
                         }
to_subtract1_old = ["rate", "residence", "num-credits", "liable"]
to_subtract1_new = ["checking-status", "savings", "employment", "rate", "sex-status", "guarantors", "residence",
                    "property", "installment", "housing", "num-credits", "job", "liable", "phone", "foreign"]

# True encodings of the dataset
ordinal_features = {"checking-status": 4, "savings": 5, "employment": 5, "rate": 4, "guarantors": 3, "residence": 4,
                    "property": 4, "housing": 3, "num-credits": 4, "job": 4, "liable": 2}
discrete_features = {"credit-history": 5, "purpose": 11, "sex-status": 4, "installment": 3, "phone": 2, "foreign": 2}
continuous_features = ["duration", "amount", "age"]
columns = ["checking-status", "duration", "credit-history", "purpose", "amount", "savings", "employment", "rate",
           "sex-status", "guarantors",
           "residence", "property", "age", "installment", "housing", "num-credits", "job", "liable", "phone", "foreign",
           "good-credit"]


def load_old(path):
    df_old = pd.read_csv(path, header=None, delimiter=',')
    df_old = df_old.dropna()
    df_old.columns = columns

    # map the true meaning of the old dataset to the new dataset's encodings (both starts from number 0)
    df_old = df_old.replace(to_replace=feature_encodings_old)
    for name in to_subtract1_old:
        df_old[name] = df_old[name] - 1

    # min max scale
    min_vals = np.min(df_old[continuous_features], axis=0)
    max_vals = np.max(df_old[continuous_features], axis=0)
    df_old_mm = min_max_scale(df_old, continuous_features, min_vals, max_vals)
    # encodings
    preprocessor_old = Preprocessor(ordinal_features, discrete_features, columns)
    df_old_enc = preprocessor_old.encode_df(df_old_mm)

    return df_old, df_old_mm, df_old_enc, preprocessor_old


def load_new(path):
    df_new = pd.read_csv(path, header=None)
    df_new = df_new.dropna()
    df_new = df_new.drop(columns=[0])  # drop ID column
    df_new = df_new.drop(labels=0)  # drop headings row
    df_new.columns = columns
    for name in df_new.columns:
        df_new[name] = df_new[name].astype(int)
    for name in to_subtract1_new:
        df_new[name] = df_new[name] - 1

    # min max scale
    min_vals = np.min(df_new[continuous_features], axis=0)
    max_vals = np.max(df_new[continuous_features], axis=0)
    df_new_mm = min_max_scale(df_new, continuous_features, min_vals, max_vals)

    # encodings
    preprocessor_new = Preprocessor(ordinal_features, discrete_features, columns)
    df_new_enc = preprocessor_new.encode_df(df_new_mm)
    df_new_enc = df_new_enc.dropna()

    return df_new, df_new_mm, df_new_enc, preprocessor_new
