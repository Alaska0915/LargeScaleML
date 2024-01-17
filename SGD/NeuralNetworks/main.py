#!/usr/bin/env python3
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy
from numpy import random
import scipy
import matplotlib
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

import torch
import torchvision
## you may wish to import other things like torch.nn
import copy
from torch.utils.data import DataLoader
import datetime

### hyperparameter settings and other constants
batch_size = 100
num_classes = 10
epochs = 10
mnist_input_shape = (28, 28, 1)
d1 = 1024
d2 = 256
alpha = 0.1
beta = 0.9
alpha_adam = 0.001
rho1 = 0.99
rho2 = 0.999
### end hyperparameter settings


# load the MNIST dataset using TensorFlow/Keras
def load_MNIST_dataset():
	train_dataset = torchvision.datasets.MNIST(
		root = './data',
		train = True,
		transform = torchvision.transforms.ToTensor(),
		download = True)
	test_dataset = torchvision.datasets.MNIST(
		root = './data',
		train = False,
		transform = torchvision.transforms.ToTensor(),
		download = False)
	splitted_train_dataset, validation_dataset = torch.utils.data.random_split(
        train_dataset, [50000, 10000], generator=torch.Generator().manual_seed(1))
	return (train_dataset, test_dataset, splitted_train_dataset, validation_dataset)

# construct dataloaders for the MNIST dataset
#
# train_dataset        input train dataset (output of load_MNIST_dataset)
# test_dataset         input test dataset (output of load_MNIST_dataset)
# batch_size           batch size for training
# shuffle_train        boolean: whether to shuffle the training dataset
#
# returns              tuple of (train_dataloader, test_dataloader)
#     each component of the tuple should be a torch.utils.data.DataLoader object
#     for the corresponding training set;
#     use the specified batch_size and shuffle_train values for the training DataLoader;
#     use a batch size of 100 and no shuffling for the test data loader
def construct_dataloaders(dataset, batch_size, shuffle_train=True):
	# TODO students should implement this
    train_dataset, test_dataset, splitted_train_dataset, validation_dataset = dataset
    train_dataloader = DataLoader(train_dataset,
                                batch_size = batch_size,
                                shuffle = shuffle_train,)
    test_dataloader = DataLoader(test_dataset,
                                batch_size = 100,
                                shuffle = False,)
    splitted_train_dataloader = DataLoader(splitted_train_dataset,
                                batch_size = batch_size,
                                shuffle = shuffle_train,)
    validation_dataloader = DataLoader(validation_dataset,
                                batch_size = 100,
                                shuffle = False,)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['test'] = test_dataloader
    dataloaders['splitted_train'] = splitted_train_dataloader
    dataloaders['validation'] = validation_dataloader
    return dataloaders


# evaluate a trained model on MNIST data
#
# dataloader    dataloader of examples to evaluate on
# model         trained PyTorch model
# loss_fn       loss function (e.g. torch.nn.CrossEntropyLoss)
#
# returns       tuple of (loss, accuracy), both python floats
@torch.no_grad()
def evaluate_model(dataloader, model, loss_fn):
	# TODO students should implement this
	loss, accuracy = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			y_hat = model(X)
			loss += loss_fn(y_hat, y).item()
			accuracy += (y_hat.argmax(1) == y).type(torch.float).sum().item()
	loss = loss / len(dataloader)
	accuracy = accuracy / len(dataloader)
	return (loss, accuracy)

# build a fully connected two-hidden-layer neural network for MNIST data, as in Part 1.1
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_fully_connected_model_part1_1(d1, d2):
	# TODO students should implement this
    model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, d1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d1, d2),
            torch.nn.ReLU(inplace=True),
			torch.nn.Linear(d2, 10),
        )

    return model

# build a fully connected two-hidden-layer neural network with Batch Norm, as in Part 1.4
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_fully_connected_model_part1_4(d1, d2):
	# TODO students should implement this
    model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, d1),
			torch.nn.BatchNorm1d(d1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d1, d2),
			torch.nn.BatchNorm1d(d2),
            torch.nn.ReLU(inplace=True),
			torch.nn.Linear(d2, 10),
			torch.nn.BatchNorm1d(10),
        )

    return model

# build a convolutional neural network, as in Part 3.1
# use the default initialization for the parameters provided in PyTorch
#
# returns   a new model of type torch.nn.Sequential
def make_cnn_model_part3_1():
	# TODO students should implement this
    model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Flatten(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 10),
        )

    return model

def make_cnn_model_part3_2(d2):
	# TODO students should implement this
    model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Flatten(),
            torch.nn.Linear(512, d2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(d2, 10),
        )

    return model

# train a neural network on MNIST data
#     be sure to call model.train() before training and model.eval() before evaluating!
#
# train_dataloader   training dataloader
# test_dataloader    test dataloader
# model              dnn model to be trained (training should mutate this)
# loss_fn            loss function
# optimizer          an optimizer that inherits from torch.optim.Optimizer
# epochs             number of epochs to run
# eval_train_stats   boolean; whether to evaluate statistics on training set each epoch
# eval_test_stats    boolean; whether to evaluate statistics on test set each epoch
#
# returns   a tuple of
#   train_loss       an array of length `epochs` containing the training loss after each epoch, or [] if eval_train_stats == False
#   train_acc        an array of length `epochs` containing the training accuracy after each epoch, or [] if eval_train_stats == False
#   test_loss        an array of length `epochs` containing the test loss after each epoch, or [] if eval_test_stats == False
#   test_acc         an array of length `epochs` containing the test accuracy after each epoch, or [] if eval_test_stats == False
#   approx_tr_loss   an array of length `epochs` containing the average training loss of examples processed in this epoch
#   approx_tr_acc    an array of length `epochs` containing the average training accuracy of examples processed in this epoch
def train(dataloaders, model, loss_fn, optimizer, epochs, eval_train_stats=True, eval_test_stats=True, require_validation=False, real_time_printing=False):
	# TODO students should implement this
    # Return a 6-tuple is making the result messy. We should all return dict here

    # Terminal Printing Control
    if real_time_printing:
        print('Train Start')

    # Load the correct dataloader
    if require_validation:
        train_dataloader = dataloaders['splitted_train']
        validation_dataloader = dataloaders['validation']
    else:
        train_dataloader = dataloaders['train']

    test_dataloader = dataloaders['test']

    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    approx_tr_loss, approx_tr_acc = [], []
    validation_loss, validation_acc = [], []
    result = {}

    for i in range(epochs):
        if real_time_printing:
            print(f'Current training in epoch {i}')
        model.train()
        running_loss, running_accuracy = 0.0, 0.0
        for X, y in train_dataloader:
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_accuracy += (y_hat.argmax(1) == y).type(torch.float).sum().item()
        running_loss /= len(train_dataloader)
        running_accuracy /= len(train_dataloader)

        approx_tr_loss.append(copy.deepcopy(running_loss))
        approx_tr_acc.append(copy.deepcopy(running_accuracy))
        if real_time_printing:
            print(f'Avergae Train Accuracy in epoch {i} is {running_accuracy}')

        model.eval()
        if eval_train_stats:
            tr_loss, tr_accuracy = evaluate_model(train_dataloader, model, loss_fn)
            train_loss.append(copy.deepcopy(tr_loss))
            train_acc.append(copy.deepcopy(tr_accuracy))
            if real_time_printing:
                print(f'Training Accuracy after epoch {i} is {tr_accuracy}')

        if eval_test_stats:
            te_loss, te_accuracy = evaluate_model(test_dataloader, model, loss_fn)
            test_loss.append(copy.deepcopy(te_loss))
            test_acc.append(copy.deepcopy(te_accuracy))
            if real_time_printing:
                print(f'Testing Accuracy after in epoch {i} is {te_accuracy}')

        if require_validation:
            va_loss, va_accuracy = evaluate_model(validation_dataloader, model, loss_fn)
            validation_loss.append(copy.deepcopy(va_loss))
            validation_acc.append(copy.deepcopy(va_accuracy))
            if real_time_printing:
                print(f'Validation Accuracy after in epoch {i} is {va_accuracy}')

    result['train_loss'] = train_loss
    result['train_acc'] = train_acc
    result['test_loss'] = test_loss
    result['test_acc'] = test_acc
    result['approx_tr_loss'] = approx_tr_loss
    result['approx_tr_acc'] = approx_tr_acc
    if require_validation:
        result['validation_loss'] = validation_loss
        result['validation_acc'] = validation_acc

    return result

# Helper function for Part 2. All use SGD + Momentum with model 1
def grid_search(dataloaders, lr_list = [alpha], momentum_list = [beta], d1_list = [d1], d2_list = [d2], real_time_printing=False):
    best_validation_accuracy = 0
    best_parameters = {}
    best_parameters_performance = {}
    overall_parameters_performance = {}
    for alpha in lr_list:
        for beta in momentum_list:
            for d1 in d1_list:
                for d2 in d2_list:
                    if real_time_printing:
                        print(f'\nCurrent working on training model with alpha={alpha}, beta={beta}, d1={d1}, d2={d2}')
                    model = make_fully_connected_model_part1_1(d1, d2)
                    optimizer = torch.optim.SGD(model.parameters(), lr = alpha, momentum = beta)
                    start_tr = datetime.datetime.now()
                    single_result = train(dataloaders = dataloaders,
                          model = model,
                          loss_fn = torch.nn.CrossEntropyLoss(),
                          optimizer = optimizer,
                          epochs = epochs,
                          eval_train_stats=False,
                          require_validation=True,)
                    single_validation_accuracy = single_result['validation_acc'][-1]
                    single_validation_loss = single_result['validation_loss'][-1]
                    single_test_accuracy = single_result['test_acc'][-1]
                    single_test_loss = single_result['test_loss'][-1]
                    end_tr = datetime.datetime.now()
                    training_time = end_tr-start_tr
                    single_result = (single_validation_accuracy, single_validation_loss, single_test_accuracy, single_test_loss, str(training_time))
                    overall_parameters_performance[(alpha, beta, d1, d2)] = single_result
                    if real_time_printing:
                        print(single_result)
                        print(f'The validation accuracy after {epochs} epochs is {single_validation_accuracy}')
                        print(f'The testing accuracy after {epochs} epochs is {single_test_accuracy}')
                        print(f"Time taken for training this model is:  {end_tr-start_tr}s")
                    if single_validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = single_validation_accuracy
                        best_parameters['alpha'] = alpha
                        best_parameters['d1'] = d1
                        best_parameters['d2'] = d2
                        best_parameters_performance['validation_accuracy'] = single_validation_accuracy
                        best_parameters_performance['validation_loss'] = single_validation_loss
                        best_parameters_performance['test_acc'] = single_test_accuracy
                        best_parameters_performance['test_loss'] = single_test_loss
    return best_parameters, best_parameters_performance, overall_parameters_performance

from scipy.stats import loguniform
def random_search(dataloaders, lr_dist, d1_dist, d2_dist, real_time_printing=False, n = 100, beta_dist = beta):
    best_validation_accuracy = 0
    best_parameters = {}
    best_parameters_performance = {}
    overall_parameters_performance = {}
    for i in range(n):
        alpha = lr_dist.rvs()
        d1 = int(d1_dist.rvs())
        d2 = int(d2_dist.rvs())
        print(f'\nCurrent working on training model with alpha={alpha}, d1={d1}, d2={d2}')
        model = make_fully_connected_model_part1_1(d1, d2)
        optimizer = torch.optim.SGD(model.parameters(), lr = alpha, momentum = beta)
        start_tr = datetime.datetime.now()
        single_result = train(dataloaders = dataloaders,
                          model = model,
                          loss_fn = torch.nn.CrossEntropyLoss(),
                          optimizer = optimizer,
                          epochs = epochs,
                          eval_train_stats=False,
                          require_validation=True,)
        single_validation_accuracy = single_result['validation_acc'][-1]
        single_validation_loss = single_result['validation_loss'][-1]
        single_test_accuracy = single_result['test_acc'][-1]
        single_test_loss = single_result['test_loss'][-1]
        end_tr = datetime.datetime.now()
        training_time = end_tr-start_tr
        single_result = (single_validation_accuracy, single_validation_loss, single_test_accuracy, single_test_loss, str(training_time))
        overall_parameters_performance[(alpha, beta, d1, d2)] = single_result
        if real_time_printing:
            print(single_result)
            print(f'The validation accuracy after {epochs} epochs is {single_validation_accuracy}')
            print(f'The testing accuracy after {epochs} epochs is {single_test_accuracy}')
            print(f"Time taken for training this model is:  {end_tr-start_tr}s")
        if single_validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = single_validation_accuracy
            best_parameters['alpha'] = alpha
            best_parameters['d1'] = d1
            best_parameters['d2'] = d2
            best_parameters_performance['validation_accuracy'] = single_validation_accuracy
            best_parameters_performance['validation_loss'] = single_validation_loss
            best_parameters_performance['test_acc'] = single_test_accuracy
            best_parameters_performance['test_loss'] = single_test_loss
    return best_parameters, best_parameters_performance, overall_parameters_performance

# Helper function to get number of parameters in a PyTorch model
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def get_parameter_num(model, trainable):
    if trainable:
        num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num =  sum(p.numel() for p in model.parameters())
    return num

if __name__ == "__main__":
    # Data Preprocess
    dataset = load_MNIST_dataset()
    dataloaders = construct_dataloaders(dataset, batch_size, shuffle_train=True)

    TEST_BREAKER = '*' * 20
    # Part 1.1: Simple SGD
    print(f"\n{TEST_BREAKER}\nPart 1.1\n")
    model_1 = make_fully_connected_model_part1_1(d1, d2)
    optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()


    start = datetime.datetime.now()
    result_1_1 = train(dataloaders = dataloaders,
      model = model_1,
      loss_fn = loss_fn,
      optimizer = optimizer,
      epochs = epochs,)
    end = datetime.datetime.now()
    print(result_1_1)
    print(f"Time taken for Part 1.1 is:  {end-start}s")
    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), result_1_1['approx_tr_loss'], label='approx_tr_loss', color='b')
    ax.plot(list(range(epochs)), result_1_1['train_loss'], label='train_loss', color='r')
    ax.plot(list(range(epochs)), result_1_1['test_loss'], label='test_loss', color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title('Result1_1 Loss Plot')
    ax.legend()
    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), result_1_1['approx_tr_acc'], label='approx_tr_acc', color='b')
    ax.plot(list(range(epochs)), result_1_1['train_acc'], label='train_acc', color='r')
    ax.plot(list(range(epochs)), result_1_1['test_acc'], label='test_acc', color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Result1_1 Acc Plot')
    ax.legend()
    plt.show()

    # Part 1.2: SGD with Momentum
    print(f"\n{TEST_BREAKER}\nPart 1.2\n")
    optimizer_2 = torch.optim.SGD(model_1.parameters(), lr = alpha, momentum = beta)
    start = datetime.datetime.now()

    result_1_2 = train(dataloaders = dataloaders,
      model = model_1,
      loss_fn = loss_fn,
      optimizer = optimizer_2,
      epochs = epochs,)
    end = datetime.datetime.now()
    print(result_1_2)
    print(f"Time taken for Part 1.2 is:  {end-start}s")
    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), result_1_2['approx_tr_loss'], label='approx_tr_loss', color='b')
    ax.plot(list(range(epochs)), result_1_2['train_loss'], label='train_loss', color='r')
    ax.plot(list(range(epochs)), result_1_2['test_loss'], label='test_loss', color='g')

    # # Add labels and a legend
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title('Result1_2 Loss Plot')
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), result_1_2['approx_tr_acc'], label='approx_tr_acc', color='b')
    ax.plot(list(range(epochs)), result_1_2['train_acc'], label='train_acc', color='r')
    ax.plot(list(range(epochs)), result_1_2['test_acc'], label='test_acc', color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Result1_2 Acc Plot')
    ax.legend()
    plt.show()

    # Part 1.3: Adam
    print(f"\n{TEST_BREAKER}\nPart 1.3\n")
    optimizer_3 = torch.optim.Adam(model_1.parameters(), lr=alpha_adam, betas=(rho1, rho2))
    start = datetime.datetime.now()
    result_1_3 = train(dataloaders = dataloaders,
      model = model_1,
      loss_fn = loss_fn,
      optimizer = optimizer_3,
      epochs = epochs,)
    end = datetime.datetime.now()
    print(result_1_3)
    print(f"Time taken for Part 1.3 is:  {end-start}s")
    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), result_1_3['approx_tr_loss'], label='approx_tr_loss', color='b')
    ax.plot(list(range(epochs)), result_1_3['train_loss'], label='train_loss', color='r')
    ax.plot(list(range(epochs)), result_1_3['test_loss'], label='test_loss', color='g')

    # # Add labels and a legend
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title('Result1_3 Loss Plot')
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), result_1_3['approx_tr_acc'], label='approx_tr_acc', color='b')
    ax.plot(list(range(epochs)), result_1_3['train_acc'], label='train_acc', color='r')
    ax.plot(list(range(epochs)), result_1_3['test_acc'], label='test_acc', color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Result1_3 Acc Plot')
    ax.legend()
    plt.show()

    # Part 1.4: BN, SGD with Momentum
    print(f"\n{TEST_BREAKER}\nPart 1.4\n")
    model_4 = make_fully_connected_model_part1_4(d1, d2)
    optimizer_4 = torch.optim.SGD(model_4.parameters(), lr = 0.001, momentum = beta)
    start = datetime.datetime.now()
    result_1_4 = train(dataloaders = dataloaders,
      model = model_4,
      loss_fn = loss_fn,
      optimizer = optimizer_4,
      epochs = epochs,)
    end = datetime.datetime.now()
    print(result_1_4)
    print(f"Time taken for Part 1.4 is:  {end-start}s")
    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), result_1_4['approx_tr_loss'], label='approx_tr_loss', color='b')
    ax.plot(list(range(epochs)), result_1_4['train_loss'], label='train_loss', color='r')
    ax.plot(list(range(epochs)), result_1_4['test_loss'], label='test_loss', color='g')

    # # Add labels and a legend
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title('Result1_4 Loss Plot')
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), result_1_4['approx_tr_acc'], label='approx_tr_acc', color='b')
    ax.plot(list(range(epochs)), result_1_4['train_acc'], label='train_acc', color='r')
    ax.plot(list(range(epochs)), result_1_4['test_acc'], label='test_acc', color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Result1_4 Acc Plot')
    ax.legend()
    plt.show()

    # Part 2.1: lr search for SGD with momentum
    print(f"\n{TEST_BREAKER}\nPart 2.1\n")
    lr_list = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    start = datetime.datetime.now()
    best_parameters, best_parameters_performance, overall_parameters_performance = grid_search(dataloaders, lr_list = lr_list, real_time_printing=True)
    end = datetime.datetime.now()
    print(best_parameters)
    print(best_parameters_performance)
    print(overall_parameters_performance)
    print(f"Time taken for Part 2.1 is:  {end-start}s")


    # Part 2.2: Grid Search
    print(f"\n{TEST_BREAKER}\nPart 2.2\n")
    lr_list = [0.1, 0.03, 0.01, 0.003]
    d1_list = [128, 256, 512, 1024, 2048]
    d2_list = [32, 64, 128, 256, 512]
    start = datetime.datetime.now()
    best_parameters, best_parameters_performance, overall_parameters_performance = grid_search(dataloaders, lr_list = lr_list, d1_list = d1_list, d2_list = d2_list, real_time_printing=True)
    end = datetime.datetime.now()
    print(best_parameters)
    print(best_parameters_performance)
    print('!!!The overall parameter performance is')
    print(overall_parameters_performance)
    print(f"Time taken for Part 2.2 is:  {end-start}s")

    # Part 2.3: Random Search
    print(f"\n{TEST_BREAKER}\nPart 2.3\n")
    lr_dist = loguniform(1e-5, 1e0)
    d1_dist = loguniform(128, 2048)
    d2_dist = loguniform(32, 512)
    start = datetime.datetime.now()
    best_parameters, best_parameters_performance, overall_parameters_performance = random_search(dataloaders, lr_dist = lr_dist, d1_dist = d1_dist, d2_dist = d2_dist, real_time_printing=True, n = 100)
    end = datetime.datetime.now()
    print(best_parameters)
    print(best_parameters_performance)
    print('!!!The overall parameter performance is')
    print(overall_parameters_performance)
    print(f"Time taken for Part 2.3 is:  {end-start}s")

    # Part 3.1: CNN + Adam
    print(f"\n{TEST_BREAKER}\nPart 3.1\n")
    model_cnn = make_cnn_model_part3_1()
    optimizer_5 = torch.optim.Adam(model_cnn.parameters(), lr=alpha_adam, betas=(rho1, rho2))
    start = datetime.datetime.now()
    result_3_1 = train(dataloaders = dataloaders,
      model = model_cnn,
      loss_fn = loss_fn,
      optimizer = optimizer_5,
      epochs = epochs,)
    end = datetime.datetime.now()
    print(result_3_1)
    print(f"Time taken for Part 3 is:  {end-start}s")
    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), result_3_1['approx_tr_loss'], label='Minibatch_tr_loss', color='b')
    ax.plot(list(range(epochs)), result_3_1['train_loss'], label='train_loss', color='r')
    ax.plot(list(range(epochs)), result_3_1['test_loss'], label='test_loss', color='g')

    # Add labels and a legend
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    ax.set_title('Result3_1 Loss Plot')
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(list(range(epochs)), result_3_1['approx_tr_acc'], label='Minibatch_tr_acc', color='b')
    ax.plot(list(range(epochs)), result_3_1['train_acc'], label='train_acc', color='r')
    ax.plot(list(range(epochs)), result_3_1['test_acc'], label='test_acc', color='g')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Result3_1 Acc Plot')
    ax.legend()
    plt.show()


    # Part 3.4: Model Parameters
    print(f"\n{TEST_BREAKER}\nPart 3.4\n")
    (d1, d2) = (1024, 256)
    model_1 = make_fully_connected_model_part1_1(d1, d2)
    model_4 = make_fully_connected_model_part1_4(d1, d2)
    model_cnn = make_cnn_model_part3_1()
    model_cnn_2 = make_cnn_model_part3_2(d2 = 64)
    parameter_count = {}

    parameter_count['Connected_Trainable'] = get_parameter_num(model = model_1, trainable = True)
    parameter_count['Connected_BN_Trainable'] = get_parameter_num(model = model_4, trainable = True)
    parameter_count['CNN_Trainable'] = get_parameter_num(model = model_cnn, trainable = True)
    parameter_count['CNN_Trainable_2'] = get_parameter_num(model = model_cnn_2, trainable = True)


    print(parameter_count)

    # Part 3.5: CNN Improvement


    # Method 1: less_parameter
    print(f"\n{TEST_BREAKER}\nPart 3.5.1\n")
    model_cnn_2 = make_cnn_model_part3_2(d2 = 64)
    optimizer_5 = torch.optim.Adam(model_cnn_2.parameters(), lr=alpha_adam, betas=(rho1, rho2))
    start = datetime.datetime.now()
    result_3_2 = train(dataloaders = dataloaders,
      model = model_cnn_2,
      loss_fn = loss_fn,
      optimizer = optimizer_5,
      epochs = epochs,)
    end = datetime.datetime.now()
    print(result_3_2)
    print(f"Time taken for Part 3.5.1 is:  {end-start}s")


    # Method 2: better optimizer
    print(f"\n{TEST_BREAKER}\nPart 3.5.2\n")
    model_cnn = make_cnn_model_part3_1()
    optimizer_6 = torch.optim.Adam(model_cnn.parameters(), lr=0.0003, betas=(rho1, rho2))
    start = datetime.datetime.now()
    result_3_3 = train(dataloaders = dataloaders,
      model = model_cnn,
      loss_fn = loss_fn,
      optimizer = optimizer_6,
      epochs = epochs,)
    end = datetime.datetime.now()
    print(result_3_3)
    print(f"Time taken for Part 3.5.2 is:  {end-start}s")


