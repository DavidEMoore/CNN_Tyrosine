import pandas as pd
import numpy as np
from operator import mul
from functools import partial
import os

def load_data():
    # load positive data and resize to add label
    os.chdir('/Users/Dave/git_tyro/Metabolite-Prediction/Tyrosine/')
    positive = pd.read_csv('positive_train.csv', header=True)
    negative = pd.read_csv('negative_train.csv', header=True)

    ppm = positive.ix[:,0] # First column are the variable names    
    train = np.concatenate((positive.ix[:,1:np.size(positive,axis=1)],negative.ix[:,1:np.size(negative,axis=1)]),axis=1) # combine them except the first column
    train = train.T
    train = pd.DataFrame(train) # Now convert it to a data frame
    train['label'] = 0 # Now add in a label column initially set to all 0's
    train.loc[0:94,'label'] = 1 # Now set the first 94 samples to 1 to indicate they are positive
    
    train.columns = ["X"+str(x) for x in list(ppm)] + ['label'] # Now correct the column names

    random_order = np.random.permutation(np.size(train,axis=0)) #sample(1:nrow(features),nrow(features))
    train = train.ix[random_order,:]

    # count rows and dims
    n = train.shape[0]

    # partition train, test, validate sets
    train_frac, test_frac, valid_frac = 0.7, 0.2, 0.1
    train_offset, test_offset, valid_offset = map(int,
                                                  map(partial(mul, n),
                                                      (train_frac, test_frac,
                                                       valid_frac)))
    train_set, test_set, valid_set = (train[0:train_offset],
                                      train[train_offset+1:
                                               train_offset+1+test_offset],
                                      train[train_offset+1+test_offset+1:
                                               train_offset+1+test_offset+1+valid_offset])

    return train_set, valid_set, test_set


def test_load_data():
    print "Loading data set ..."
    train, valid, test = load_data()
    print "Checking size ..."
    assert train.shape == (723, 1095)
    assert valid.shape == (103, 1095)
    assert test.shape == (206, 1095)
    print "Done."


if __name__ == '__main__':
    test_load_data()
