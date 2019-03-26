<a name="Parameters"></a>
# Parameters

Our model is based on different parameters that can be changed and executed by the `main.py` file.

* GPU parameters:

  * `gpu`: Enables cuda, 1 for enabling and 0 for disabling. (`default: 1`);
  * `gpuID`: If one has multiple-GPUs, you can switch the default GPU. The GPU IDs are 1-indexed, so using GPU 3 can be specified as `-gpuID 3`. (`default: 0`).


* Training parameters:
  * `batch_size`: Batch size used in training. (`default: 200`);
  * `num_epochs`: Number of epochs to consider during training. (`default: 100`);
  * `decay_epoch`: Reduces the learning rate every `decay_epoch`. Used in *step decay* of the learning rate. For instance, `-decayEpoch 10` reduces the learning rate by `-lr_decay` every 10 epochs. (`default: -1`);
  * `learning_rate`: Initial learning rate for training. (`default: 0.001`);
  * `lr_decay`: Decreases the learning rate by this value, it is used jointly with `decay_epoch`. (`default: 0.5`);
  * `pretrain`: Number of iterations to pretrain the model with the variational autoencoder losses only. (`default: -1`).


* Architecture parameters:

  * `num_classes`: Number of classes. (`default: 10`);
  * `feature_size`: Size of the deterministic feature vector learnt by the network. (`default: 150`);
  * `gaussian_size`: Size of the gaussian latent vector learnt by the network. (`default: 100`);


* Partition parameters:

  * `train_proportion`: Percentage of data to consider for training. The rest is considered for validation. Range of values [0.0 - 1.0]. (`default: 0.8`);
  * `batch_size_val`: Batch size used in validation data. (`default: 200`);
  * `batch_size_test`: Batch size used in test data. (`default: 200`);

  For a partition with 80% for training and 20% for validation we specify `-train 0.8`.


* Gumbel parameters:

  * `temperature`: Initial temperature used in gumbel-softmax. Recommended range of values [0.5-1.0]. (`default: 1.0`);
  * `decay_temperature`: Set 1 to decay gumbel temperature every epoch. (`default: 0`);
  * `hard_gumbel`: Hard version of gumbel-softmax in forward propagation. (`default: 0`);
  * `min_temperature`: Minimum temperature value after annealing. (`default: 0.5`);
  * `decay_temp_rate`: Rate of temperature decay every epoch. (`default: 0.00693`).

  Temperature decay is based on *Exponential decay* -> ``newTemperature = temperature * exp(-decayTempRate * currentEpoch)``. The value of `decayTempRate` can be obtained from that formulation according to the number of epochs used in training.

* Loss function parameters:

  * `loss_type`: Desired loss function to train the model (mse, bce). (`default: bce`);
  * `w_gaussian`: Weight of the gaussian regularization loss. (`default: 1.0`);
  * `w_categorical`: Weight of the categorical regularization loss. (`default: 1.0`);
  * `w_reconstruction`: Weight of the reconstruction loss. (`default: 1.0`);
  * `w_metric`: Weight of the metric distance loss. (`default: 1.0`);
  * `w_assign`: Weight of the assignment loss. (`default: 1.0`);
  * `metric_margin`: Margin of metric loss. (`default: 0.5`);
  * `metric_loss`: Desired metric loss function to train (triplet, lifted). (`default: lifted`).


* Semisupervised parameters:

  * `num_labeled`: Number of labeled data to consider from the input dataset. This number must be divisible by the `num_classes` parameter because we will get balanced number of labeled data per class. For instance, a value of `-num_labeled 1000 -num_classes 10` gets 100 labeled samples of each category/class. (`default:100`);
  * `num_labeled_batch`: Number of labeled data to consider per batch. For instance, a value of `-num_labeled_batch 50 -batch_size 200` the batch will contain 50 labeled samples and 150 unlabeled samples. It is useful when considering large size of labeled samples and small batch size. (`default:100`);
  * `knn`: Number of neighbors to consider in the assignment of categories. (`default: 1`).


* Other parameters:

  * `dataset`: Dataset to use. It can be mnist or svhn. (`default: mnist`);
  * `seed`: Seed used to initialize random weights and partitions. Set -1 for random seed. (`default: -1`);
  * `verbose`: Print additional details while training. (`default: 0`).
