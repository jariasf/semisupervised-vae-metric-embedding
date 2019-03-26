"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Semisupervised generative model with metric embedding auxiliary task

"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from networks.CatVAENetwork import *
from losses.LossFunctions import *
from utils.partition import *
from utils.assignment import *

class SSVAE:

    def __init__(self, params):
      self.batch_size = params.batch_size
      self.batch_size_val = params.batch_size_val
      self.initial_temperature = params.temperature
      self.decay_temperature = params.decay_temperature
      self.num_epochs = params.num_epochs
      self.loss_type = params.loss_type
      self.num_classes = params.num_classes
      self.w_gauss = params.w_gaussian
      self.w_categ = params.w_categorical
      self.w_recon = params.w_reconstruction
      self.decay_temp_rate = params.decay_temp_rate
      self.gaussian_size = params.gaussian_size
      self.feature_size = params.feature_size
      self.min_temperature = params.min_temperature
      self.temperature = params.temperature # current temperature
      self.verbose = params.verbose
      
      self.sess = tf.Session()
      self.network = CatVAENetwork(params)
      self.losses = LossFunctions()

      self.w_assign = params.w_assign
      self.num_labeled = params.num_labeled
      self.knn = params.knn
      self.metric_loss = params.metric_loss
      
      self.w_metric = tf.placeholder(tf.float32, [])
      self.initial_w_metric = params.w_metric
      self._w_metric = params.w_metric
      self.anneal_metric_loss = params.anneal_w_metric
      
      self.learning_rate = tf.placeholder(tf.float32, [])
      self.lr = params.learning_rate
      self.decay_epoch = params.decay_epoch
      self.lr_decay = params.lr_decay
      
      self.pretrain = params.pretrain
      self.num_labeled_batch = params.num_labeled_batch
      self.dataset = params.dataset
      
      self.metric_margin = params.metric_margin


    def create_dataset(self, is_training, data, labels, batch_size, x_labeled = None, y_labeled = None):
      """Create dataset given input data

      Args:
          is_training: (bool) whether to use the train or test pipeline.
                       At training, we shuffle the data and have multiple epochs
          data: (array) corresponding array containing the input data
          labels: (array) corresponding array containing the labels of the input data
          batch_size: (int) size of each batch to consider from the data
          x_labeled: (array) corresponding array containing the labeled input data
          y_labeled: (array) corresponding array containing the labeles of the labeled input data
 
      Returns:
          output: (dict) contains what will be the input of the tensorflow graph
      """
      num_samples = data.shape[0]
      
      # create dataset object      
      if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(data)
      else:
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))

      # shuffle data in training phase
      if is_training:  
        dataset = dataset.shuffle(num_samples).repeat()

      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(1)

      # create reinitializable iterator from dataset
      iterator = dataset.make_initializable_iterator()

      labeled_data = False
      if labels is None:
        data = iterator.get_next()
      else:
        data, labels = iterator.get_next()
        
        # append labeled data to each batch
        if x_labeled is not None:
          labeled_data = True
          _data = tf.concat([data, x_labeled], 0)
          _labels = tf.concat([labels, y_labeled], 0)
          
      iterator_init = iterator.initializer
      
      if labeled_data:
        output = {'data': _data, 'labels': _labels, 'iterator_init': iterator_init}
        output['labels_semisupervised'] = y_labeled
      else:
        output = {'data': data, 'labels': labels, 'iterator_init': iterator_init}
        output['labels_semisupervised'] = None
      return output


    def create_model(self, is_training, inputs, output_size):
      """Model function defining the graph operations.

      Args:
          is_training: (bool) whether we are in training phase or not
          inputs: (dict) contains the inputs of the graph (features, labels...)
                  this can be `tf.placeholder` or outputs of `tf.data`
          output_size: (int) size of the output layer

      Returns:
          model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
      """
      data, _labels = inputs['data'], inputs['labels']

      # create network and obtain latent vectors that will be used in loss functions
      latent_spec = self.network.encoder(data, self.num_classes, is_training)      

      gaussian, mean, logVar  = latent_spec['gaussian'], latent_spec['mean'], latent_spec['logVar']
      categorical, prob, log_prob = latent_spec['categorical'], latent_spec['prob_cat'], latent_spec['log_prob_cat']
      _logits, features = latent_spec['logits'], latent_spec['features']

      output = self.network.decoder(gaussian, categorical, output_size, is_training)
      
      # reconstruction loss
      if self.loss_type == 'bce':
        loss_rec = self.losses.binary_cross_entropy(data, output)
      elif self.loss_type == 'mse':
        loss_rec = tf.losses.mean_squared_error(data, output)
      else:
        raise "invalid loss function... try bce or mse..."
      
      # kl-divergence loss
      loss_kl = self.losses.kl_gaussian(mean, logVar)
      loss_kl_cat = self.losses.kl_categorical(prob, log_prob, self.num_classes)
      
      # auxiliary task to assign labels and regularize the feature space
      if _labels is not None:
        labeled_ss = inputs['labels_semisupervised']
        if labeled_ss is not None:
          # assignment loss only if labeled data is available (training phase)
          predicted_labels = assign_labels_semisupervised(features, labeled_ss, self.num_labeled_batch, 
                                                          self.batch_size, self.num_classes, self.knn)
          # use assigned labels and logits to calculate cross entropy loss
          loss_assign = tf.losses.sparse_softmax_cross_entropy(labels=predicted_labels, logits=_logits)
        else:
          # predict labels from logits or softmax(logits) (validation/testing phase)
          loss_assign = tf.constant(0.)
          predicted_labels = tf.argmax(prob, axis=1)
        
        # calculate accuracy using the predicted and true labels
        accuracy = tf.reduce_mean( tf.cast( tf.equal(_labels, predicted_labels), tf.float32 ) )
        
        # metric embedding loss
        if self.metric_loss == 'triplet':
          loss_metric = tf.contrib.losses.metric_learning.triplet_semihard_loss(predicted_labels, features, margin=self.metric_margin)
        elif self.metric_loss == 'lifted':
          loss_metric = tf.contrib.losses.metric_learning.lifted_struct_loss(predicted_labels, features, margin=self.metric_margin)
        else:
          raise "invalid metric loss... currently we support triplet and lifted loss"

      else:
        accuracy = tf.constant(0.)
        loss_assign = tf.constant(0.)
        loss_metric = tf.constant(0.)
        predicted_labels = tf.constant(0.)

      # variational autoencoder loss
      loss_vae = self.w_recon * loss_rec
      loss_vae += self.w_gauss * loss_kl
      loss_vae += self.w_categ * loss_kl_cat

      # total loss
      loss_total = loss_vae + self.w_assign * loss_assign + self.w_metric * loss_metric
      
      if is_training:
        # use adam for optimization
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        
        # needed for batch normalization layer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op_vae = optimizer.minimize(loss_vae)
          train_op_tot = optimizer.minimize(loss_total)
      
      # create model specification
      model_spec = inputs
      model_spec['variable_init_op'] = tf.global_variables_initializer()
      model_spec['output'] = output
      model_spec['features'] = features
      model_spec['predicted_labels'] = predicted_labels
      model_spec['true_labels'] = _labels
      model_spec['loss_rec'] = loss_rec      
      model_spec['loss_kl'] = loss_kl      
      model_spec['loss_kl_cat'] = loss_kl_cat
      model_spec['loss_total'] = loss_total
      model_spec['loss_metric'] = loss_metric
      model_spec['loss_assign'] = loss_assign
      model_spec['accuracy'] = accuracy
      
      # optimizers are only available in training phase
      if is_training:
        model_spec['train_op'] = train_op_tot
        model_spec['train_vae'] = train_op_vae
      
      return model_spec
    

    def evaluate_dataset(self, is_training, num_batches, model_spec, labeled_data = None, labeled_labels = None):
      """Evaluate the model

      Args:
          is_training: (bool) whether we are training or not
          num_batches: (int) number of batches to train/test
          model_spec: (dict) contains the graph operations or nodes needed for evaluation
          labeled_data: (array) corresponding array containing the labeled input data
          labeled_labels: (array) corresponding array containing the labeles of the labeled input data

      Returns:
          (dic) average of loss functions and metrics for the given number of batches
      """
      avg_accuracy = 0.0
      avg_nmi = 0.0
      avg_loss_rec = 0.0
      avg_loss_kl = 0.0
      avg_loss_cat = 0.0
      avg_loss_total = 0.0
      avg_loss_metric = 0.0
      avg_loss_assign = 0.0
      
      # initialize dataset iteratior
      self.sess.run(model_spec['iterator_init'])
      
      if is_training:
        
        # pretraining will train only the variational autoencoder losses
        if self.pretrain < 1:
          train_optimizer = model_spec['train_op']
        else:
          train_optimizer = model_spec['train_vae']
        
        # training phase
        for j in range(num_batches):
          # select randomly subsets of labeled data according to the batch size
          _x_labeled, _y_labeled, _, _ = create_semisupervised_dataset(labeled_data, labeled_labels, 
                                                                       self.num_classes, self.num_labeled_batch)
          
          # run the tensorflow flow graph
          _, loss_rec, loss_kl, loss_metric, loss_assign, loss_cat, loss_total, accuracy = self.sess.run([train_optimizer, 
                                                     model_spec['loss_rec'], model_spec['loss_kl'],
                                                     model_spec['loss_metric'], model_spec['loss_assign'], 
                                                     model_spec['loss_kl_cat'], model_spec['loss_total'],
                                                     model_spec['accuracy']],
                                                     feed_dict={self.network.temperature: self.temperature
                                                                , self.w_metric: self._w_metric
                                                                , self.learning_rate: self.lr
                                                                , self.x_labeled: _x_labeled
                                                                , self.y_labeled: _y_labeled})

          # accumulate values
          avg_accuracy += accuracy
          avg_loss_rec += loss_rec
          avg_loss_kl += loss_kl
          avg_loss_cat += loss_cat
          avg_loss_total += loss_total
          avg_loss_metric += loss_metric
          avg_loss_assign += loss_assign
      else:
        # validation phase
        for j in range(num_batches):
          # run the tensorflow flow graph
          loss_rec, loss_kl, loss_metric, loss_assign, loss_cat, loss_total, accuracy = self.sess.run([ 
                                                     model_spec['loss_rec'], model_spec['loss_kl'], 
                                                     model_spec['loss_metric'], model_spec['loss_assign'], 
                                                     model_spec['loss_kl_cat'], model_spec['loss_total'],
                                                     model_spec['accuracy']],
                                                     feed_dict={self.network.temperature: self.temperature
                                                                ,self.w_metric: self._w_metric
                                                                ,self.learning_rate: self.lr})     

          # accumulate values
          avg_accuracy += accuracy
          avg_loss_rec += loss_rec
          avg_loss_kl += loss_kl
          avg_loss_cat += loss_cat
          avg_loss_total += loss_total
          avg_loss_metric += loss_metric
          avg_loss_assign += loss_assign

      # average values by the given number of batches
      avg_loss_rec /= num_batches
      avg_loss_kl /= num_batches
      avg_accuracy /= num_batches
      avg_loss_cat /= num_batches
      avg_loss_total /= num_batches
      avg_loss_metric /= num_batches
      avg_loss_assign /= num_batches
      
      return {'avg_loss_rec': avg_loss_rec, 'avg_loss_kl': avg_loss_kl, 'avg_loss_cat': avg_loss_cat, 
              'total_loss': avg_loss_total, 'avg_accuracy': avg_accuracy, 
              'avg_loss_metric': avg_loss_metric, 'avg_loss_assign': avg_loss_assign}


    def train(self, train_data, train_labels, val_data, val_labels, labeled_data, labeled_labels):
      """Train the model

      Args:
          train_data: (array) corresponding array containing the training data
          train_labels: (array) corresponding array containing the labels of the training data
          val_data: (array) corresponding array containing the validation data
          val_labels: (array) corresponding array containing the labels of the validation data
          labeled_data: (array) corresponding array containing the labeled input data
          labeled_labels: (array) corresponding array containing the labeles of the labeled input data

      Returns:
          output: (dict) contains the history of train/val loss
      """
      train_history_loss, val_history_loss = [], []
      train_history_acc, val_history_acc = [], []
      train_history_nmi, val_history_nmi = [], []
      
      # placeholders for the labeled data
      self.x_labeled = tf.placeholder(tf.float32, shape = [self.num_labeled_batch, labeled_data.shape[1]])
      self.y_labeled = tf.placeholder(tf.int64, shape = [self.num_labeled_batch])
      
      # create training and validation dataset
      train_dataset = self.create_dataset(True, train_data, train_labels, 
                                          self.batch_size - self.num_labeled_batch, self.x_labeled, self.y_labeled)
      val_dataset = self.create_dataset(True, val_data, val_labels, self.batch_size_val)
      
      self.output_size = train_data.shape[1]
    
      # create train and validation models      
      train_model = self.create_model(True, train_dataset, self.output_size)
      val_model = self.create_model(False, val_dataset, self.output_size)
    
      # set number of batches
      num_train_batches = int(np.ceil(train_data.shape[0] / (1.0 * (self.batch_size - self.num_labeled_batch))))
      num_val_batches = int(np.ceil(val_data.shape[0] / (1.0 * self.batch_size_val)))

      # initialize global variables
      self.sess.run( train_model['variable_init_op'] )

      # training and validation phases
      print('Training phase...')
      for i in range(self.num_epochs):

        # pretraining at each epoch
        if self.pretrain > 0:
          self.pretrain = self.pretrain - 1
        
        # decay learning rate according to decay_epoch parameter
        if self.decay_epoch > 0 and (i + 1) % self.decay_epoch == 0:
          self.lr = self.lr * self.lr_decay
          print('Decaying learning rate: %lf' % self.lr)
        
        # evaluate train and validation datasets
        train_loss = self.evaluate_dataset(True, num_train_batches, train_model, labeled_data, labeled_labels)
        val_loss = self.evaluate_dataset(False, num_val_batches, val_model)
       
        # get training results for printing
        train_loss_rec = train_loss['avg_loss_rec']
        train_loss_kl = train_loss['avg_loss_kl']
        train_loss_cat = train_loss['avg_loss_cat']
        train_loss_ass = train_loss['avg_loss_assign']
        train_loss_met = train_loss['avg_loss_metric']
        train_accuracy = train_loss['avg_accuracy']
        train_total_loss = train_loss['total_loss']
        # get validation results for printing
        val_loss_rec = val_loss['avg_loss_rec']
        val_loss_kl = val_loss['avg_loss_kl']
        val_loss_cat = val_loss['avg_loss_cat']
        val_loss_ass = val_loss['avg_loss_assign']
        val_loss_met = val_loss['avg_loss_metric']
        val_accuracy = val_loss['avg_accuracy']
        val_total_loss = val_loss['total_loss']        

        # if verbose then print specific information about training
        if self.verbose == 1:
          print("(Epoch %d / %d) REC=Train: %.5lf; Val: %.5lf  KL=Train: %.5lf; Val: %.5lf  KL-Cat=Train: %.5lf; Val: %.5lf  MET=Train: %.5lf; Val: %.5lf  ASS=Train: %.5lf; Val: %.5lf  ACC=Train %.5lf; Val %.5lf" % \
                (i + 1, self.num_epochs, train_loss_rec, val_loss_rec, train_loss_kl, val_loss_kl, train_loss_cat, val_loss_cat, train_loss_met, val_loss_met, train_loss_ass, val_loss_ass, train_accuracy, val_accuracy))
        else:
          print("(Epoch %d / %d) Train Loss: %.5lf; Val Loss: %.5lf   Train Accuracy: %.5lf; Val Accuracy: %.5lf" % \
                (i + 1, self.num_epochs, train_total_loss, val_total_loss, train_accuracy, val_accuracy))
        
        # save loss and accuracy of each epoch
        train_history_loss.append(train_total_loss)
        val_history_loss.append(val_total_loss)
        train_history_acc.append(train_accuracy)
        val_history_acc.append(val_accuracy)
        
        if self.anneal_metric_loss == 1:
          #anneal loss from initial_w_metric to 1 in the first 100 epochs
          self._w_metric = np.minimum(self.initial_w_metric * np.exp(0.06908*(i+1)),1)
          if self.verbose == 1:
            print('Metric Weight: %.5lf' % self._w_metric)
        
        if self.decay_temperature == 1:
          # decay temperature of gumbel-softmax
          self.temperature = np.maximum(self.initial_temperature*np.exp(-self.decay_temp_rate*(i + 1) ),self.min_temperature)
          if self.verbose == 1:
            print("Gumbel Temperature: %.5lf" % self.temperature)

      return {'train_history_loss' : train_history_loss, 'val_history_loss': val_history_loss,
              'train_history_acc': train_history_acc, 'val_history_acc': val_history_acc}
    
    
    def test(self, test_data, test_labels, batch_size = -1):
      """Test the model with new data

      Args:
          test_data: (array) corresponding array containing the testing data
          test_labels: (array) corresponding array containing the labels of the testing data
          batch_size: (int) batch size used to run the model
          
      Return:
          accuracy for the given test data

      """
      # if batch_size is not specified then use all data
      if batch_size == -1:
        batch_size = test_data.shape[0]
        
      # create dataset
      test_dataset = self.create_dataset(False, test_data, test_labels, batch_size)
      true_labels = test_dataset['labels']
      
      # perform a forward call on the encoder to obtain predicted labels
      latent = self.network.encoder(test_dataset['data'], self.num_classes)
      logits = latent['prob_cat']
      predicted_labels = tf.argmax(logits, axis=1)
      
      # calculate accuracy given the predicted and true labels
      accuracy = tf.reduce_mean( tf.cast( tf.equal(true_labels, predicted_labels), tf.float32 ) )
      
      # initialize dataset iterator
      self.sess.run(test_dataset['iterator_init'])
      
      # calculate number of batches given batch size
      num_batches = int(np.ceil(test_data.shape[0] / (1.0 * batch_size)))
      
      # evaluate the model
      avg_accuracy = 0.0
      for j in range(num_batches):
        _accuracy = self.sess.run(accuracy,
                                   feed_dict={self.network.temperature: self.temperature
                                             ,self.w_metric: self._w_metric
                                             ,self.learning_rate: self.lr})
        avg_accuracy += _accuracy
      
      # average the accuracy
      avg_accuracy /= num_batches
      return avg_accuracy
    

    def latent_features(self, data, batch_size=-1):
      """Obtain latent features learnt by the model

      Args:
          data: (array) corresponding array containing the data
          batch_size: (int) size of each batch to consider from the data

      Returns:
          features: (array) array containing the features from the data
      """
      # if batch_size is not specified then use all data
      if batch_size == -1:
        batch_size = data.shape[0]
      
      # create dataset  
      dataset = self.create_dataset(False, data, None, batch_size)

      # we will use only the encoder network
      latent = self.network.encoder(dataset['data'], self.num_classes)
      encoder = latent['features']
      
      # obtain the features from the input data
      self.sess.run(dataset['iterator_init'])      
      num_batches = data.shape[0] // batch_size
      features = np.zeros((data.shape[0], self.feature_size))
      for j in range(num_batches):
        features[j*batch_size:j*batch_size + batch_size] = self.sess.run(encoder,
                                                                        feed_dict={self.network.temperature: self.temperature
                                                                                  ,self.w_metric: self._w_metric
                                                                                  ,self.learning_rate: self.lr})
      return features
    
    
    def reconstruct_data(self, data, batch_size=-1):
      """Reconstruct Data

      Args:
          data: (array) corresponding array containing the data
          batch_size: (int) size of each batch to consider from the data

      Returns:
          reconstructed: (array) array containing the reconstructed data
      """
      # if batch_size is not specified then use all data
      if batch_size == -1:
        batch_size = data.shape[0]
      
      # create dataset
      dataset = self.create_dataset(False, data, None, batch_size)

      # reuse model used in training
      model_spec = self.create_model(False, dataset, data.shape[1])

      # obtain the reconstructed data
      self.sess.run(model_spec['iterator_init'])      
      num_batches = data.shape[0] // batch_size      
      reconstructed = np.zeros(data.shape)
      pos = 0
      for j in range(num_batches):
        reconstructed[pos:pos + batch_size] = self.sess.run(model_spec['output'],
                                                            feed_dict={self.network.temperature: self.temperature
                                                                      ,self.w_metric: self._w_metric
                                                                      ,self.learning_rate:self.lr})
        pos += batch_size
      return reconstructed
    

    def plot_latent_space(self, data, labels, save=False):
      """Plot the latent space learnt by the model

      Args:
          data: (array) corresponding array containing the data
          labels: (array) corresponding array containing the labels
          save: (bool) whether to save the latent space plot

      Returns:
          fig: (figure) plot of the latent space
      """
      # obtain the latent features
      features = self.latent_features(data)
      
      # plot only the first 2 dimensions
      fig = plt.figure(figsize=(8, 6))
      plt.scatter(features[:, 0], features[:, 1], c=labels, marker='o',
              edgecolor='none', cmap=plt.cm.get_cmap('jet', 10), s = 10)
      plt.colorbar()
      if(save):
          fig.savefig('latent_space.png')
      return fig
    

    def generate_data(self, num_elements=1, category=0):
      """Generate data for a specified category

      Args:
          num_elements: (int) number of elements to generate
          category: (int) category from which we will generate data

      Returns:
          generated data according to num_elements
      """
      # gaussian noise for each element
      noise = tf.random_normal([num_elements, self.gaussian_size],mean = 0, stddev = 1, dtype= tf.float32)
      indices = (np.ones(num_elements)*category).astype(int).tolist()
      # category is specified with a one-hot array
      categorical = tf.one_hot(indices, self.num_classes)
      # use the gaussian noise and category to generate data from the generator
      out = self.network.decoder(noise, categorical, self.output_size)
      return self.sess.run(out, feed_dict={self.network.temperature: self.temperature
                                          ,self.w_metric: self._w_metric
                                          ,self.learning_rate:self.lr})
    

    def random_generation(self, num_elements=1):
      """Random generation for each category

      Args:
          num_elements: (int) number of elements to generate

      Returns:
          generated data according to num_elements
      """
      # gaussian noise for each element
      noise = tf.random_normal([num_elements * self.num_classes, self.gaussian_size],
                                mean = 0, stddev = 1, dtype= tf.float32)
      # categories for each element
      arr = np.array([])
      for i in range(self.num_classes):
        arr = np.hstack([arr,np.ones(num_elements) * i] )
      indices = arr.astype(int).tolist()
      categorical = tf.one_hot(indices, self.num_classes)
      # use the gaussian noise and categories to generate data from the generator
      out = self.network.decoder(noise, categorical, self.output_size)
      return self.sess.run(out, feed_dict={self.network.temperature: self.temperature
                                          ,self.w_metric: self._w_metric
                                          ,self.learning_rate:self.lr})
    
    
    def style_generation(self, data):
      """Style transfer generation for each category given a predefined style

      Args:
          data: (array)  corresponding array containing the input style

      Returns:
          generated data according to the style of data
      """
      # convert data to tensor
      num_elem = data.shape[0]
      data = np.repeat(data, self.num_classes, axis=0)
      tf_data = tf.convert_to_tensor(data)
      # get latent gaussian features from the encoder
      latent = self.network.encoder(tf_data, self.num_classes)
      gaussian = latent['gaussian']
      # set one-hot values for each category
      indices = np.tile(range(self.num_classes), num_elem)
      categorical = tf.one_hot(indices, self.num_classes)
      # use the gaussian features and categories to generate data from the generator
      out = self.network.decoder(gaussian, categorical, self.output_size)
      return self.sess.run(out, feed_dict={self.network.temperature: self.temperature
                                          ,self.w_metric: self._w_metric
                                          ,self.learning_rate:self.lr})
      
