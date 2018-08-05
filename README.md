# Andrea Valenti's Restricted Boltzmann Machine implementation

This project provides a MATLAB implementation of a [Restricted Boltzmann Machine](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) (RBM for short) neural network.

## Features

This project
- Implements the RBM model.
- Applies the RBM to the well-known [MNIST dataset](http://yann.lecun.com/exdb/mnist/) in order to learn a lower-dimensional implementation for the handwritten digits images.

## Getting Started

- Files *loadMNISTImages.m* and *loadMNISTLabels.m* contain helper functions used to load the MNIST data.
- File *one_hot.m* contains a helper function used to convert a scalar integer to a [*one hot* representation](https://en.wikipedia.org/wiki/One-hot).
- Files *rbm_init.m* *rbm_train.m* *rbm_encode.m* *rbm_gradient.m* *rbm_update.m* and *rbm_contrastive_divergence.m* implement the various parts of the RBM.
- File *rbm.m* is the "main" script. After training the RBM over the MNIST digits, it encode those digits in a lower dimensional representation (basically, the hidden units' activation). Then, it trains a [softmax layer](https://en.wikipedia.org/wiki/Softmax_function#Neural_networks) to classify those encoded digits representations. After the training, a number of plots with useful information (training error epochs, Training/Test confusion matrices, hidden units' learned weights) is showed.

**NB**: The MNIST dataset is provided in the *dataset.zip* compressed archive. You need to unzip the archive first in order to successfully run the code.

### Installation/Dependencies

You just need a MATLAB installation. Version R2016b or later ones are fine (however it probably works with previous versions as well).

### Usage

With the current hyperparameters setting, you should be able to achieve about 93% accuracy over the test set (not bad, considering we have about 90% of compression rate, from 1024 input pixels down to 100 hidden units!).

You are ancouraged to try different hyperparameters combinations. The most important parameters are:
- Number of hidden units. (!)
- Number of training epochs. (!)
- Learning rate.
- Momentum rate.
- Regularization (weight decay) rate. 

**NB**: parameters marked with (!) have a significant impact on the complexity of the model, higher values can really slow down the training process.

You might also want to vary the number of Contrastive-Divergence steps to perform for each training sample, however I don't particuarly recommend it, since in practice it doesn't seem to have a high impact on the quality of the result, while it can significantly slow down the training process.

## Getting Help

In the "references" subfolder it is provided the [excellent tutorial by Geoffrey Hinton on RBM training](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf). For any theoretical doubt, you should refer to it, I don't think you could find better advices elsewhere.

For more code-related issues, or any other additional information, you can email me at valentiandrea@rocketmail.com.

## Contributing Guidelines  

You're free (and encouraged) to make your own contributions to this project.

## Code of Conduct

Just be nice, and respecful of each other!

## License

All source code from this project is released under the MIT license.