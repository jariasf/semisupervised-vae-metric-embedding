"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Categorical Variational Autoencoder Networks

"""

import tensorflow as tf

class CatVAENetwork:
    eps = 1e-8
    
    def __init__(self, params):
      if params is not None:
        self.temperature = tf.placeholder(tf.float32, [])
        self.feature_size = params.feature_size
        self.gaussian_size = params.gaussian_size
        self.hard_gumbel = params.hard_gumbel
        self.loss_type = params.loss_type
        self.dataset = params.dataset    

    def latent_gaussian(self, hidden, gaussian_size):
      """Sample from the Gaussian distribution
      
      Args:
        hidden: (array) [batch_size, n_features] features obtained by the encoder
        gaussian_size: (int) size of the gaussian sample vector
        
      Returns:
        (dict) contains the nodes of the mean, log of variance and gaussian
      """
      out = hidden
      mean = tf.layers.dense(out, units=gaussian_size)
      logVar = tf.layers.dense(out, units=gaussian_size)
      noise = tf.random_normal(tf.shape(mean), mean = 0, stddev = 1, dtype= tf.float32)
      z = mean + tf.sqrt(tf.exp(logVar) + self.eps) * noise
      return {'mean': mean, 'logVar': logVar, 'gaussian': z}
    

    def sample_gumbel(self, shape):
      """Sample from Gumbel(0, 1)
      
      Args:
         shape: (array) containing the dimensions of the specified sample
      """
      U = tf.random_uniform(shape, minval=0, maxval=1)
      return -tf.log(-tf.log(U + self.eps) + self.eps)


    def gumbel_softmax(self, logits, temperature, hard=False):
      """Sample from the Gumbel-Softmax distribution and optionally discretize.
      
      Args:
        logits: (array) [batch_size, n_class] unnormalized log-probs
        temperature: (float) non-negative scalar
        hard: (boolean) if True, take argmax, but differentiate w.r.t. soft sample y
        
      Returns:
        y: (array) [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
      """
      gumbel_softmax_sample = logits + self.sample_gumbel(tf.shape(logits))
      y = tf.nn.softmax(gumbel_softmax_sample / self.temperature)
      if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
      return y


    def encoder_conv32x32(self, input_data, num_classes, is_training=False):
      """Convolutional inference network for 32x32x3 input images
      
      Args:
        input_data: (array) [batch_size, n_features=3072] input images
        num_classes: (int) number of classification classes
        is_training: (bool) whether we are in training phase or not

      Returns:
        (dict) contains the features, gaussian and categorical information
      """
      with tf.variable_scope('encoder_conv32x32', reuse=not is_training):
        out = input_data
        out = tf.reshape(out, [-1, 32, 32, 3])
        
        # number of filters, kernels and stride
        filters = [16, 32, 64, 128, 256, self.feature_size]
        kernels = [5, 5, 5, 3, 4, 8]
        stride = [1, 1, 1, 1, 2, 1]

        # encoding from input image to deterministic features
        for i, num_filters in enumerate(filters):
          out = tf.layers.conv2d(out, num_filters, kernel_size=kernels[i], strides=stride[i],
                                 padding="valid")
          out = tf.layers.batch_normalization(out, training=is_training)
          out = tf.nn.relu(out)
        out = tf.layers.flatten(out)

        # defining layers to learn the gaussian distribution
        gaussian = self.latent_gaussian(out, self.gaussian_size)

        # defining layers to learn the categorical distribution
        logits = tf.layers.dense(out, units=num_classes)
        categorical = self.gumbel_softmax(logits, self.temperature, self.hard_gumbel)
        prob = tf.nn.softmax(logits)
        log_prob = tf.log(prob + self.eps)

        # keep graph output operations that will be used in loss functions
        output = gaussian
        output['categorical'] = categorical
        output['prob_cat'] = prob
        output['log_prob_cat'] = log_prob
        output['features'] = out
        output['logits'] = logits
        return output
    

    def decoder_conv32x32(self, gaussian, categorical, output_size, is_training=False):
      """Convolutional generative network for 32x32x3 input images
      
      Args:
        gaussian: (array) [batch_size, gaussian_size] latent gaussian vector
        categorical: (array) [batch_size, num_classes] latent categorical vector
        output_size: (int) size of the output image
        is_training: (bool) whether we are in training phase or not

      Returns:
        (array) array containing the generated/reconstructed image
      """
      with tf.variable_scope('decoder_conv32x32', reuse=not is_training):

        # convert gaussian and categorical to same shape
        gaussian = tf.layers.dense(gaussian, units=self.feature_size)
        categorical = tf.layers.dense(categorical, units=self.feature_size)

        # add categorical and gaussian vectors
        out = gaussian + categorical
        
        # reshape vector for convolutional layers
        out = tf.reshape(out, [-1, 1, 1, self.feature_size])

        # number of filters, kernels and stride
        filters = [256, 128, 64, 32, 16, 3]
        kernels = [8, 4, 3, 5, 5, 5]
        strides = [1, 2, 1, 1, 1, 1]
        
        # decoding from categorical and gaussian to output image
        for i, num_filters in enumerate(filters):
          out = tf.layers.conv2d_transpose(out, num_filters, kernel_size=kernels[i], strides=strides[i], 
                                 padding='valid')
          out = tf.layers.batch_normalization(out, training=is_training)
          out = tf.nn.relu(out)
        out = tf.reshape(out, [-1, 32 * 32 * 3])
        
        # define output layer according to loss function
        if self.loss_type == 'bce':
          out = tf.layers.dense(out, units=output_size, activation=tf.nn.sigmoid)
        else:
          out = tf.layers.dense(out, units=output_size)
        return out
    

    def encoder_conv(self, input_data, num_classes, is_training=False):
      """Convolutional inference network for 28x28x1 input images
      
      Args:
        input_data: (array) [batch_size, n_features=784] input images
        num_classes: (int) number of classification classes
        is_training: (bool) whether we are in training phase or not

      Returns:
        (dict) contains the features, gaussian and categorical information
      """
      with tf.variable_scope('encoder_cnn', reuse=not is_training):
        out = input_data
        out = tf.reshape(out, [-1, 28, 28, 1])

        # number of filters, kernels and stride
        filters = [16, 32, 64, 128, self.feature_size]
        kernels = [5, 5, 4, 5, 3]
        stride = [1, 1, 2, 2, 1]

        # encoding from input image to deterministic features
        for i, num_filters in enumerate(filters):
          out = tf.layers.conv2d(out, num_filters, kernel_size=kernels[i], strides=stride[i],
                                 padding="valid")
          out = tf.layers.batch_normalization(out, training=is_training)
          out = tf.nn.relu(out)
        out = tf.layers.flatten(out)

        # defining layers to learn the gaussian distribution
        gaussian = self.latent_gaussian(out, self.gaussian_size)

        # defining layers to learn the categorical distribution
        logits = tf.layers.dense(out, units=num_classes)
        categorical = self.gumbel_softmax(logits, self.temperature, self.hard_gumbel)
        prob = tf.nn.softmax(logits)
        log_prob = tf.log(prob + self.eps)
        
        # keep graph output operations that will be used in loss functions
        output = gaussian
        output['categorical'] = categorical
        output['prob_cat'] = prob
        output['log_prob_cat'] = log_prob
        output['features'] = out
        output['logits'] = logits
        return output
    

    def decoder_conv(self, gaussian, categorical, output_size, is_training=False):
      """Convolutional generative network for 28x28x1 input images
      
      Args:
        gaussian: (array) [batch_size, gaussian_size] latent gaussian vector
        categorical: (array) [batch_size, num_classes] latent categorical vector
        output_size: (int) size of the output image
        is_training: (bool) whether we are in training phase or not

      Returns:
        (array) array containing the generated/reconstructed image
      """
      with tf.variable_scope('decoder_conv', reuse=not is_training):

        # convert gaussian and categorical to same shape
        gaussian = tf.layers.dense(gaussian, units=self.feature_size)
        categorical = tf.layers.dense(categorical, units=self.feature_size)

        # add categorical and gaussian vectors
        out = gaussian + categorical

        # reshape vector for convolutional layers
        out = tf.reshape(out, [-1, 1, 1, self.feature_size])
        
        # number of filters, kernels and stride
        filters = [128, 64, 32, 16, 1]
        kernels = [3, 5, 4, 5, 5]
        strides = [1, 2, 2, 1, 1]

        # decoding from categorical and gaussian to output image
        for i, num_filters in enumerate(filters):
          out = tf.layers.conv2d_transpose(out, num_filters, kernel_size=kernels[i], strides=strides[i], 
                                 padding='valid')
          out = tf.layers.batch_normalization(out, training=is_training)
          out = tf.nn.relu(out)
        out = tf.reshape(out, [-1, 28 * 28])

        # define output layer according to loss function
        if self.loss_type == 'bce':
          out = tf.layers.dense(out, units=output_size, activation=tf.nn.sigmoid)
        else:
          out = tf.layers.dense(out, units=output_size)
        return out    


    def encoder_fc(self, input_data, num_classes, is_training=False):
      """Fully connected inference network
      
      Args:
        input_data: (array) [batch_size, n_features=784] input images
        num_classes: (int) number of classification classes
        is_training: (bool) whether we are in training phase or not

      Returns:
        (dict) contains the features, gaussian and categorical information
      """
      with tf.variable_scope('encoder_fc', reuse=not is_training):
        out = input_data
        
        # encoding from input image to deterministic features
        out = tf.layers.dense(out, units=500)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.dense(out, units=500)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.dense(out, units=self.feature_size)
        
        # defining layers to learn the gaussian distribution
        gaussian = self.latent_gaussian(out, self.gaussian_size)

        # defining layers to learn the categorical distribution
        logits = tf.layers.dense(out, units=num_classes)
        categorical = self.gumbel_softmax(logits, self.temperature, self.hard_gumbel)
        prob = tf.nn.softmax(logits)
        log_prob = tf.log(prob + self.eps)
        
        # keep graph output operations that will be used in loss functions
        output = gaussian
        output['categorical'] = categorical
        output['prob_cat'] = prob
        output['log_prob_cat'] = log_prob
        output['features'] = out
        output['logits'] = logits
      return output
    

    def decoder_fc(self, gaussian, categorical, output_size, is_training=False):
      """Fully connected generative network
      
      Args:
        gaussian: (array) [batch_size, gaussian_size] latent gaussian vector
        categorical: (array) [batch_size, num_classes] latent categorical vector
        output_size: (int) size of the output image
        is_training: (bool) whether we are in training phase or not

      Returns:
        (array) array containing the generated/reconstructed image
      """
      with tf.variable_scope('decoder_fc', reuse=not is_training):
        
        # convert gaussian and categorical to same shape
        gaussian = tf.layers.dense(gaussian, units=self.feature_size)
        categorical = tf.layers.dense(categorical, units=self.feature_size)
        
        # add categorical and gaussian vectors
        out = gaussian + categorical
        
        # decoding from categorical and gaussian to output image
        out = tf.layers.dense(out, units=self.feature_size)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.dense(out, units=500)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.relu(out)     
        out = tf.layers.dense(out, units=500)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.relu(out)

        # define output layer according to loss function
        if self.loss_type == 'bce':
          out = tf.layers.dense(out, units=output_size, activation=tf.nn.sigmoid)
        else:
          out = tf.layers.dense(out, units=output_size)
      return out  


    def encoder(self, input_data, num_classes, is_training=False):
      """Inference/Encoder network
      
      Args:
        input_data: (array) [batch_size, n_features] input images
        num_classes: (int) number of classification classes
        is_training: (bool) whether we are in training phase or not

      Returns:
        (dict) contains the features, gaussian and categorical information
      """
      if self.dataset == 'mnist':
        # for the mnist dataset we use the 28x28x1 convolutional network
        latent_spec = self.encoder_conv(input_data, num_classes, is_training)
      else:
        # for the svhn dataset we use the 32x32x3 convolutional network
        latent_spec = self.encoder_conv32x32(input_data, num_classes, is_training)
      return latent_spec
    
    
    def decoder(self, gaussian, categorical, output_size, is_training=False):
      """Generative/Decoder network of our model
      
      Args:
        gaussian: (array) [batch_size, gaussian_size] latent gaussian vector
        categorical: (array) [batch_size, num_classes] latent categorical vector
        output_size: (int) size of the output image
        is_training: (bool) whether we are in training phase or not

      Returns:
        (array) array containing the generated/reconstructed image
      """
      if self.dataset == 'mnist':
        # for the mnist dataset we use the 28x28x1 convolutional network
        output = self.decoder_conv(gaussian, categorical, output_size, is_training)
      else:
        # for the svhn dataset we use the 32x32x3 convolutional network
        output = self.decoder_conv32x32(gaussian, categorical, output_size, is_training)
      return output
