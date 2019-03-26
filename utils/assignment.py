"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Util functions for assignment of unlabeled data

"""

import tensorflow as tf

def bincount_matrix(x, num_classes):
  """Count number of occurrences of each value in array of non-negative values. 
  
  Args:
      x: (array) corresponding array containing non-negative values
      num_classes: (int) number of classification classes

  Returns:
      output: (array) corresponding array containing number of occurrences
  """
  x = tf.cast(x, tf.int32)
  max_x_plus_1 = tf.constant(num_classes, dtype=tf.int32)
  ids = x + max_x_plus_1*tf.range(tf.shape(x)[0])[:,None]
  out = tf.reshape(tf.bincount(tf.layers.flatten(ids), 
                 minlength=max_x_plus_1*tf.shape(x)[0]), [-1, num_classes])
  return out


def assign_labels_semisupervised(features, labels, num_labeled, batch_size, num_classes, knn):
  """Assign labels to unlabeled data based on the k-nearest-neighbors
  
  Args:
      features: (array) corresponding array containing the features of the input data
      labels: (array) corresponding array containing the labels of the labeled data
      num_labeled: (int) num of labeled data per batch
      batch_size: (int) training batch size
      num_classes: (int) number fo classification classes
      knn: (int) number of k-nearest neighbors to use

  Returns:
      output: (array) corresponding array containing the labels assigned to all the data
  """
  dot_product = tf.matmul(features, tf.transpose(features))
  # get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
  # this also provides more numerical stability (the diagonal of the result will be exactly 0).
  # shape (batch_size,)
  square_norm = tf.diag_part(dot_product)
  # compute the pairwise distance matrix as we have:
  # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
  # shape (batch_size, batch_size)
  distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
  # because of computation errors, some distances might be negative so we put everything >= 0.0
  distances = tf.maximum(distances, 0.0)
  # get distances of unlabeled data w.r.t. labeled data
  distances = tf.slice(distances, [0, batch_size - num_labeled], [batch_size - num_labeled, -1])
  # negate distances
  neg_one = tf.constant(-1.0, dtype=tf.float32)
  neg_distances = tf.multiply(distances, neg_one)
  # get top K largest distances, because we negated the distances, we will get the closest ones
  _, idx = tf.nn.top_k(neg_distances, knn)
  # get the true labels of the K-nearest neighbors
  knn_labels = tf.gather(labels, idx)
  # count repeated labels
  count = bincount_matrix(knn_labels, num_classes)
  # assign the label of the maximum obtained from k-nn (majority vote)
  assignment = tf.argmax(count, axis=1)
  # return the assigned labels for the unlabeled data and labels of the labaled data
  return tf.concat([assignment, labels], 0)

