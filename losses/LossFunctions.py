# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Loss functions used for training our model

"""

import tensorflow as tf

class LossFunctions:
    eps = 1e-8
    
    def binary_cross_entropy(self, real, predictions, average=True):
      """Binary Cross Entropy between the true and predicted outputs
         loss = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

      Args:
          real: (array) corresponding array containing the true labels
          predictions: (array) corresponding array containing the predicted labels
          average: (bool) whether to average the result to obtain a value
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = -tf.reduce_sum( real * tf.log(predictions + self.eps) + 
                           (1 - real) * tf.log(1 - predictions + self.eps), axis = 1 )
      if average:
        return tf.reduce_mean(loss)
      else:
        return tf.reduce_sum(loss)


    def mean_squared_error(self, real, predictions, average=True):
      """Mean Squared Error between the true and predicted outputs
         loss = (1/n)*Σ(real - predicted)^2

      Args:
          real: (array) corresponding array containing the true labels
          predictions: (array) corresponding array containing the predicted labels
          average: (bool) whether to average the result to obtain a value
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = tf.square(real - predictions)
      if average:
        return tf.reduce_mean(loss)
      else:
        return tf.reduce_sum(loss)    


    def kl_gaussian(self, mean, logVar, average=True):
      """KL Divergence between the posterior and a prior gaussian distribution (N(0,1))
         loss = (1/n) * -0.5 * Σ(1 + log(σ^2) - σ^2 - μ^2)

      Args:
          mean: (array) corresponding array containing the mean of our inference model
          logVar: (array) corresponding array containing the log(variance) of our inference model
          average: (bool) whether to average the result to obtain a value
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = -0.5 * tf.reduce_sum(1 + logVar - tf.exp(logVar) - tf.square(mean + self.eps), 1 ) 
      if average:
        return tf.reduce_mean(loss)
      else:
        return tf.reduce_sum(loss)


    def kl_categorical(self, qx, log_qx, k, average=True):
      """KL Divergence between the posterior and a prior uniform distribution (U(0,1))
         loss = (1/n) * Σ(qx * log(qx/px)), because we use a uniform prior px = 1/k 
         loss = (1/n) * Σ(qx * (log(qx) - log(1/k)))

      Args:
          qx: (array) corresponding array containing the probs of our inference model
          log_qx: (array) corresponding array containing the log(probs) of our inference model
          k: (int) number of classes
          average: (bool) whether to average the result to obtain a value
 
      Returns:
          output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
      """
      loss = tf.reduce_sum(qx * (log_qx - tf.log(1.0/k)), 1)
      if average:
        return tf.reduce_mean(loss)
      else:
        return tf.reduce_sum(loss)
