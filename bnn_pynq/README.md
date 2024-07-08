# Finite-dimensions Neural networks training

This repo contains a Pytorch implementation for training networks for the paper [Architectural Optimisation in Deep Neural Networks. Tests of a theoretically inspired method.].
by [Sacha Cormenier], [Gianluca Manzan], [Paolo Branchini] and [Pierluigi Contucci].

---

This repo aims to train and test finite-dimensions neural networks. Following the work described in the article, we train and test networks while modifying the configurations of the architecture, changing the number of neurons in the hidden layers.

# Requirements

* Python >= 3.8.
* [Pytorch](https://pytorch.org) >= 1.9.1, <= 2.1 (more recent versions would be untested).
* Windows, Linux or macOS.
* GPU training-time acceleration (*Optional* but recommended).
* Install Brevitas
```bash
pip install brevitas
```

# Code

Files:
In "bnn_pynq_train.py" are all the options necessary for modifying the parameters
In "trainer.py" is the code with all the functions and classes used to make the NN train and test
In "automation_weights.sh" is the code for performing a study of different architectures

Folders:
"cfg" contains the .ini file with the architecture wanted
"exp" contains the folders where are gathered all the informations for each training and testing sessions. They are sorted by date and time of starting computation
"images" contain the different datasets "Fashion", "MNIST" and "MNIST_OddEven". The datasets must be unzipped
"models" has python functions for the use of the code

Files in each "exp" folder:
"loss_test.txt" is the loss value for the last testing session done
"loss_training.txt" gathers all the loss values during the training
"matrix_noisy.txt" gets the matrix of all the predictions made by the network during the testing session
"table_noisy" summarizes the output predictions vector and final predictions for each of the tested image
"test.txt" summarizes the functions used, the architecture, and the values for each weight and bias from the trained network

example of a command for running a training session:
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --epochs 500 --experiments exp --gpus None
you can modify options, following "bnn_pynq_train.py" arguments

For the testing: 
BREVITAS_JIT=1 python3 bnn_pynq_train.py --network GENERAL --experiments exp --gpus None --evaluate --resume exp/yourfolder/checkpoints/best.tar

# Obtain results presented in the paper

```bash
./automation_weights.sh
```


