#!/usr/bin/env python3

# Import libraries
import numpy as np
import pandas as pd

DATA_DIR = 'data/'

train_file = DATA_DIR + 'train.csv'
test_file = DATA_DIR + 'test.csv'


def convert_to_numpy(df):
    print(df)


def load_test_data():
    return pd.read_csv(test_file, header=0)

def load_train_data():
    df = pd.read_csv(train_file, header=0)
    return df.iloc[:,:-1], df.iloc[:,-1]


if __name__ == "__main__":
    dataX, datay = load_train_data()
    print(dataX.head())
    print(datay.head())
