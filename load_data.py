import pandas as pd
import numpy as np
from operator import mul
from functools import partial
import os
import theano
import numpy
import theano.tensor as T

def load_data():
    # load positive data and resize to add label
#    positive = pd.read_csv('../anderson_tyro/Metabolite-Prediction/Tyrosine/positive_train.csv', header=True)
#    negative = pd.read_csv('../anderson_tyro/Metabolite-Prediction/Tyrosine/negative_train.csv', header=True)
    positive = pd.read_csv('positive_train.csv', header=True)
    negative = pd.read_csv('negative_train.csv', header=True)

    ppm = positive.ix[:,0] # First column are the variable names    
    train = np.concatenate((positive.ix[:,1:np.size(positive,axis=1)],negative.ix[:,1:np.size(negative,axis=1)]),axis=1) # combine them except the first column
    train = train.T
    train = pd.DataFrame(train) # Now convert it to a data frame
    train['label'] = 0
    train.columns = ["X"+str(x) for x in list(ppm)] + ['label'] # Now correct the column names
    #train.ix[:,train.shape[1]-1] = 0
    train.ix[0:94,train.shape[1]-1] = 1 # Now set the first 94 samples to 1 to indicate they are positive
    #print train.ix[0:94,train.shape[1]-1]
    
    #train = pd.DataFrame(train.ix[0:200,:])
    #print train
    #print train.shape
    random_order = np.random.permutation(train.shape[0]) #sample(1:nrow(features),nrow(features))
    #print random_order
    train = train.ix[random_order,:]
    #print train.ix[0:94,train.shape[1]-1]
    #lkajdsf = ljdf
    

    # count rows and dims
    n = train.shape[0]

    # partition train, test, validate sets
    train_frac, test_frac, valid_frac = 0.7, 0.2, 0.1
    train_offset, test_offset, valid_offset = map(int,
                                                  map(partial(mul, n),
                                                      (train_frac, test_frac,
                                                       valid_frac)))
    train_set, test_set, valid_set = (train[0:train_offset].as_matrix(),
                                      train[train_offset+1:
                                               train_offset+1+test_offset].as_matrix(),
                                      train[train_offset+1+test_offset+1:
                                               train_offset+1+test_offset+1+valid_offset].as_matrix())

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        #data_x, data_y = data_xy[:,0:data_xy.shape[1]-1],data_xy[:,data_xy.shape[1]-1]
        data_x, data_y = data_xy[:,0:784],data_xy[:,data_xy.shape[1]-1]
	print data_y
	#data_x = data_x.todense()
	#print type(data_x)# == numpy.ndarray
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


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
