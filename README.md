# Semi-supervised Learning using Deep Generative Models and Metric Embedding Auxiliary Task

This project is a tensorflow implementation of my ongoing work for semi-supervised learning with deep generative models and metric embedding.

### Dependencies

1. [Tensorflow](https://www.tensorflow.org/). We tested our method with the **1.13.1** tensorflow version. You can Install Tensorflow by following the instructions on its website: [https://www.tensorflow.org/install/pip?lang=python2](https://www.tensorflow.org/install/pip?lang=python2).

*  **Caveat**: Tensorflow released the 2.0 version with different changes that will not allow to execute this implementation directly. Check the [migration guide](https://www.tensorflow.org/alpha/guide/migration_guide) for executing this implementation in the 2.0 tensorflow version.

2. [Python 2.7](https://www.python.org/downloads/). We implemented our method with the **2.7** version. Additional libraries include: numpy, scipy and matplotlib.

### Docker

* We used [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for ubuntu to train and test our models. The official tensorflow image can be obtained from [Docker Hub](https://hub.docker.com/r/tensorflow/tensorflow). The **1.13.1** version can be downloaded by using the following command:

  ```bash
  $ docker pull tensorflow/tensorflow:1.13.1-gpu
  ```

* **Note**: The official tensorflow image does not include *matplotlib* and *keras*. You will have to install them inside your container using the *pip install* command.


### Instructions

* To execute a single run of the semi-supervised model for a specific dataset:
  ```bash
  $ ./run.sh <dataset-name>
  ```
where dataset-name can be mnist or svhn.

* To change parameter values,
  ```bash
  $ ./run.sh <dataset-name> -num_epochs 200 -seed 50 -verbose 1
  ```
More details about the parameters can be found in the [parameters](parameters.md) file.

To reproduce the results from our paper, you can run these models multiple times and then average the results. By default random seeds will be used, optionally you can specify different manual seeds by setting the ``-seed`` parameter.

### Datasets

Tested with the following datasets: MNIST and SVHN. Details about each dataset can be found in the [README](dataset/README.md) file located in the *dataset* folder.

### Citation

If you find our code useful in your researches, please consider citing:

    @report{Arias2019,
        author = {Arias, Jhosimar},
        title = {Semi-supervised Learning using Deep Generative Models and Metric Embedding Auxiliary Task},
        year = {2019}
    }
