#!/bin/bash

# Switch to script directory
cd `dirname -- "$0"`

if [ -z "$1" ]; then
  echo "Please enter dataset, e.g. ./run.sh dataset_name"
  exit 0
else
  DATA=$1
  shift
fi

if [ "$DATA" == "mnist" ]; then
    python main.py -dataset mnist -gpuID 0 -num_classes 10 -batch_size 200 -batch_size_val 200 -decay_temperature 1 -learning_rate 1e-2 -seed -1 -feature_size 200 -gaussian_size 150 -w_gaussian 1.1 -w_categorical 1.2 -w_reconstruction 1 -w_metric 0.85 -w_assign 4.7 -num_labeled 100 -num_labeled_batch 100 -knn 7 -metric_loss lifted -decay_epoch 30 -lr_decay 0.5 -train_proportion 1.0 -num_epochs 100 -loss_type bce "$@"
elif [ "$DATA" == "svhn" ]; then
    python main.py -dataset svhn -gpuID 0 -num_classes 10 -batch_size 400 -batch_size_val 200 -decay_temperature 1 -learning_rate 1e-3 -seed -1 -feature_size 200 -gaussian_size 150 -w_gaussian 4.9 -w_categorical 1.5 -w_reconstruction 1 -w_metric 0.95 -w_assign 3.2 -num_labeled 1000 -num_labeled_batch 200 -knn 5 -metric_loss lifted -decay_epoch 30 -lr_decay 0.5 -metric_margin 0.9 -decay_temp_rate 0.013863 -train_proportion 1.0 -num_epochs 100 -loss_type mse "$@"
    echo "Invalid dataset. Please enter mnist or svhn"
fi

