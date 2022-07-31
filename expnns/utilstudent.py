import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessor import Preprocessor, min_max_scale

feature_encodings = {
    'school': {'GP': int(0), 'MS': int(1)},
    'sex': {'F': int(0), 'M': int(1)},
    'address': {'R': int(0), 'U': int(1)},
    'famsize': {'GT3': int(0), 'LE3': int(1)},
    'Pstatus': {'A': int(0), 'T': int(1)},
    'Mjob': {'at_home': int(0), 'health': int(1), 'other': int(2), 'services': int(3), 'teacher': int(4)},
    'Fjob': {'at_home': int(0), 'health': int(1), 'other': int(2), 'services': int(3), 'teacher': int(4)},
    'reason': {'course': int(0), 'home': int(1), 'other': int(2), 'reputation': int(3)},
    'guardian': {'father': int(0), 'mother': int(1), 'other': int(2)},
    'schoolsup': {'no': int(0), 'yes': int(1)},
    'famsup': {'no': int(0), 'yes': int(1)},
    'paid': {'no': int(0), 'yes': int(1)},
    'activities': {'no': int(0), 'yes': int(1)},
    'nursery': {'no': int(0), 'yes': int(1)},
    'higher': {'no': int(0), 'yes': int(1)},
    'internet': {'no': int(0), 'yes': int(1)},
    'romantic': {'no': int(0), 'yes': int(1)}
}
to_subtract1 = ['traveltime', 'studytime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']
continuous_features = ["age", "absences"]
discrete_features = {"sex": 2, "address": 2, "famsize": 2, "Pstatus": 2, "Mjob": 5, "Fjob": 5, "reason": 4,
                     "guardian": 3,
                     "schoolsup": 2, "famsup": 2, "paid": 2, "activities": 2, "nursery": 2, "higher": 2, "internet": 2,
                     "romantic": 2}
ordinal_features = {"Medu": 5, "Fedu": 5, "traveltime": 4, "studytime": 4, "failures": 4, "famrel": 5, "freetime": 5,
                    "goout": 5, "Dalc": 5,
                    "Walc": 5, "health": 5}
columns = ['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason',
           'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
           'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G3']


def load_dataset(path="../datasets/student/student-por.csv"):
    df = pd.read_csv(path, delimiter=';')
    df = df.dropna()
    output = np.array(df["G3"])
    df = df.drop(columns=["G1", "G2", "G3"])
    new_output = (output >= 12).astype(np.int)
    df_new_out = pd.DataFrame(data=new_output, columns=["G3"])
    df = pd.concat([df, df_new_out], axis=1)

    # map the true meaning of the old dataset to the new dataset's encodings (both starts from number 0)
    df = df.replace(to_replace=feature_encodings)
    for name in to_subtract1:
        df[name] = df[name] - 1
    df1 = df[df["school"] == 0]
    df2 = df[df["school"] == 1]
    df1 = df1.drop(columns="school")
    df2 = df2.drop(columns="school")
    # min max scale
    min_vals = np.min(df[continuous_features], axis=0)
    max_vals = np.max(df[continuous_features], axis=0)
    df1_mm = min_max_scale(df1, continuous_features, min_vals, max_vals)
    df2_mm = min_max_scale(df2, continuous_features, min_vals, max_vals)
    df2_mm = pd.DataFrame(df2_mm.values, columns=df2_mm.columns)
    # encodings
    preprocessor = Preprocessor(ordinal_features, discrete_features, columns)
    df1_enc = preprocessor.encode_df(df1_mm)
    preprocessor = Preprocessor(ordinal_features, discrete_features, columns)
    df2_enc = preprocessor.encode_df(df2_mm)
    df1_enc = pd.DataFrame(df1_enc.values[1:], columns=df1_enc.columns)
    df2_enc = pd.DataFrame(df2_enc.values[1:], columns=df2_enc.columns)

    # get X, y
    X1, y1 = df1_enc.drop(columns=['G3']), pd.DataFrame(df1_enc['G3'])
    SPLIT = .2
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, stratify=y1, test_size=SPLIT, shuffle=True,
                                                            random_state=0)
    X2, y2 = df2_enc.drop(columns=['G3']), pd.DataFrame(df2_enc['G3'])
    return X1, y1, X1_train, X1_test, y1_train, y1_test, X2, y2, preprocessor
