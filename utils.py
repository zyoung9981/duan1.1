import numpy as np
import pandas as pd


def read_data(input_path, debug=True):

    df = pd.read_csv(input_path, nrows=250 if debug else None)
    #X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].as_matrix()
    X = df.loc[:, [x for x in df.columns.tolist() if x != 'NDX']].values
    y = np.array(df.NDX)

    return X, y
