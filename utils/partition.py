"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Util functions for partitioning input data

"""

import numpy as np

def partition_train_val(x_train, y_train, proportion, num_classes, shuffle=True):
    """
    Partition data in train and validation

    Args:
        x_train: (array) corresponding array containing the input data
        y_train: (array) corresponding array containing the labels of the input data
        proportion: (float) proportion of examples to consider in the train dataset (0.0-1.0)
        num_classes: (int) number of labels

    Returns:
        train_data: (array) corresponding array containing partitioned train data
        train_labels: (array) corresponding array containing partitioned labels of train data
        val_data: (array) corresponding array containing partitioned validation data
        val_labels: (array) corresponding array containing partitioned labels of validation data
    """
    # initialize numpy arrays
    train_data_indices = np.array([], dtype=np.int32)
    val_data_indices = np.array([], dtype=np.int32)

    # iterate over the number of classes
    for i in range(0, num_classes):
        # get indices of a specific class
        subdata = np.where(y_train == i)[0]
        num_samples = subdata.shape[0]
        # randomly partition the indices based on specified proportion
        indices = np.random.permutation(num_samples)
        train_size = int(proportion * num_samples)
        train_indices, val_indices = indices[:train_size], indices[train_size:]
        # get partitioned indices
        train_subdata, val_subdata = subdata[train_indices], subdata[val_indices]
        # concatenate indices of all classes
        train_data_indices = np.hstack([train_data_indices, train_subdata])
        val_data_indices = np.hstack([val_data_indices, val_subdata])

    if shuffle:
      np.random.shuffle(train_data_indices)
      np.random.shuffle(val_data_indices)

    # get new data based on the partitioned proportions
    train_data, train_labels = x_train[train_data_indices], y_train[train_data_indices]
    val_data, val_labels = x_train[val_data_indices], y_train[val_data_indices]
    return train_data, train_labels, val_data, val_labels


def create_semisupervised_dataset(x, y, num_classes, num_labeled=100):
    """
    Partition data into labeled and unlabeled data given labeled size per class

    Args:
        x_train: (array) corresponding array containing the input data
        y_train: (array) corresponding array containing the labels of the input data
        num_classes: (int) number of classes
        num_labeled: (float) number of examples to consider in the labeled dataset

    Returns:
        x_labeled: (array) corresponding array containing partitioned labeled data
        y_labeled: (array) corresponding array containing partitioned labels of labeled data
        x_unlabaled: (array) corresponding array containing partitioned unlabeled data
        y_unlabeled: (array) corresponding array containing partitioned labels of unlabeled data
    """
    if num_labeled % num_classes != 0:
      raise "wished number of labeled samples not divisible by number of classes"
    
    # get number of labeled data per class (balanced partition)
    labeled_per_class = num_labeled//num_classes
    x_labeled = [0] * num_classes
    y_labeled = [0] * num_classes
    x_unlabeled = [0] * num_classes
    y_unlabeled = [0] * num_classes
    
    # iterate over the number of classes
    for i in range(num_classes):
        # get indices of a specific class
        indices = np.where(y == i)[0]
        # randomly partition the indices based on specified proportion
        indices = np.random.permutation(indices)
        x_labeled[i] = x[indices[:labeled_per_class]]
        y_labeled[i] = y[indices[:labeled_per_class]]
        x_unlabeled[i] = x[indices[labeled_per_class:]]
        y_unlabeled[i] = y[indices[labeled_per_class:]]    
 
    return np.vstack(x_labeled), np.hstack(y_labeled), np.vstack(x_unlabeled), np.hstack(y_unlabeled)  


def flatten_array(x):
    """
    Flatten to 2D array

    Args:
        x: (array) corresponding array containing data

    Returns:
        flatten: (array) corresponding array containing the flatten data
    """
    shape = np.prod(x.shape[1:])
    return x.reshape(-1,shape)
