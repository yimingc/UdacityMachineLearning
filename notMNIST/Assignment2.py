# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import Assignment1 as A1
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from numpy import float32

# re-format image sets to flat matrix and label to one-hot encoding
def reformat(dataset, labels, image_size, num_labels):
    dataset = np.reshape((-1, image_size * image_size)).astype(float32) # cast to float 32
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = ( np.arange(num_labels) == labels[:,None]).astype(float32) # 'None' adds a new axis
    return dataset, labels

def main():
    pickle_file = 'notMNIST.pickle'
    
    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
    
    # clean the overlaps
    #test_dataset_clean, test_dataset_keep = A1.checkOverlap(train_dataset, test_dataset)
    #valid_dataset_clean, valid_dataset_keep = A1.checkOverlap(train_dataset, valid_dataset)
    #test_labels_clean = test_labels[test_dataset_keep]
    #valid_labels_clean = valid_labels[valid_dataset_keep]
    #test_size_clean = len(test_dataset_clean)
    #valid_size_clean = len(valid_dataset_clean)
    #print('Train data size %d, label size %d.' % (len(train_dataset), len(train_labels)))
    #print('Cleaned test data size %d, label size %d.' % (test_size_clean, len(test_labels_clean)))
    #print('Cleaned valid data size %d, label size %d.' % (valid_size_clean, len(valid_labels_clean)))
        
    #Reformat into a shape that's more adapted to the models we're going to train:
    #*data as a flat matrix,
    #*labels as float 1-hot encodings.
    image_size = 28
    num_labels = 10
    train_dataset, train_labels = reformat(train_dataset, train_labels, image_size, num_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, image_size, num_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels, image_size, num_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    
    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.
    train_subset = 10000
    
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        # Load the training, validation and test data into constants that
        # are attached to the graph
        tf_train_dataset = tf.constant( train_dataset[:train_subset, :] )
        tf_train_labels = tf.constant( train_labels[:train_subset, :] )
        tf_valid_dataset = tf.constant( valid_dataset[:valid_subset, :] )
        tf_test_dataset = tf.constant( test_dataset[:test_subset, :] )
        
        # Variables.
        # These are the parameters that we are going to be training.
        # The
    
    