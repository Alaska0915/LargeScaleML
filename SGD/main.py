import os
import numpy as np
from numpy import random
import scipy
from scipy.special import softmax
import mnist
import pickle

# you can use matplotlib for plotting
import matplotlib
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

#mnist_data_directory = "data"
# TODO add any additional imports and global variables
import copy
import datetime


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_tr = np.ascontiguousarray(Xs_tr)
        Ys_tr = np.ascontiguousarray(Ys_tr)
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the cross-entropy loss of the classifier
#
# x         examples          (d)
# y         labels            (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    yhat = np.log(softmax(np.dot(W,x)))
    loss = -np.dot(y.T, yhat)
    loss += (gamma/2)*(np.linalg.norm(W))**2

    return loss

# compute the gradient of a single example of the multinomial logistic regression objective, with regularization
#
# x         training example   (d)
# y         training label     (c)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the gradient of the loss with respect to the model parameters W
def multinomial_logreg_grad_i(x, y, gamma, W):
    # TODO students should implement this in Part 1
    return np.matmul((softmax(np.matmul(W,x)) - y).reshape(-1,1), x.reshape(1,-1)) + gamma * W

# compute the error of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should implement this
    d, n = Xs.shape
    yhat = np.argmax(np.dot(W,Xs), axis=0)
    y = np.argmax(Ys, axis=0)
    err = len(np.argwhere(yhat - y))
    return err/n

# compute the gradient of the multinomial logistic regression objective on a batch, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# ii        indices of the batch (an iterable or range)
#
# returns   the gradient of the model parameters
def multinomial_logreg_batch_grad(Xs, Ys, gamma, W, ii = None, fast_version = True):
    if ii is None:
        ii = range(Xs.shape[1])
    # TODO students should implement this
    Xs = Xs[:,ii]
    Ys = Ys[:,ii]
    (d, n) = Xs.shape

    if fast_version:
        y_hat = softmax(np.dot(W,Xs), axis=0)
        del_L = np.dot(y_hat - Ys, Xs.T)
        grad = del_L + gamma * W
        return grad / n

    else:
        # a starter solution using an average of the example gradients
        acc = W * 0.0
        for i in ii:
            acc += multinomial_logreg_grad_i(Xs[:, i], Ys[:, i], gamma, W)
        return acc / len(ii)



# compute the cross-entropy loss of the classifier on a batch, with regularization
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
# ii        indices of the batch (an iterable or range)
#
# returns   the model cross-entropy loss
def multinomial_logreg_batch_loss(Xs, Ys, gamma, W, ii = None, fast_version = True):
    if ii is None:
        ii = range(Xs.shape[1])
    # TODO students should implement this
    # a starter solution using an average of the example gradients
    Xs = Xs[:,ii]
    Ys = Ys[:,ii]
    (d, n) = Xs.shape

    if fast_version:
        yhat = np.log(softmax(np.dot(W,Xs), axis=0))
        loss = -np.sum(np.multiply(Ys, yhat))
        loss += (gamma/2) * (np.linalg.norm(W))**2
        return loss/n

    else:
        acc = 0.0
        for i in ii:
            acc += multinomial_logreg_loss_i(Xs[:, i], Ys[:, i], gamma, W)
        return acc / len(ii)


# run gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs            training examples (d * n)
# Ys            training labels   (d * c)
# gamma         L2 regularization constant
# W0            the initial value of the parameters (c * d)
# alpha         step size/learning rate
# num_iters     number of iterations to run
# monitor_freq  how frequently to output the parameter vector
#
# returns       a list of models parameters, one every "monitor_freq" iterations
#               should return model parameters before iteration 0, iteration monitor_freq, iteration 2*monitor_freq, and again at the end
#               for a total of (num_iters/monitor_freq)+1 models, if num_iters is divisible by monitor_freq.
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_iters, monitor_freq, fast_version = True):
    # TODO students should implement this
    parameter = []
    for i in range(num_iters):
        if (i % monitor_freq == 0):
            parameter.append(copy.deepcopy(W0))
        W0 -= alpha * multinomial_logreg_batch_grad(Xs, Ys, gamma, W0, fast_version = fast_version)
    parameter.append(W0)
    return parameter


# ALGORITHM 1: run stochastic gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters vectors, one every "monitor_period" batches
#                   to do this, you'll want code like the following:
#                     models = []
#                     models.append(W0.copy())   # (you may not need the copy if you don't mutate W0)
#                     ...
#                     for sgd_iteration in ... :
#                       ...
#                       # code to compute a single SGD update step here
#                       ...
#                       if (it % monitor_period == 0):
#                         models.append(W)
def sgd_minibatch(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    parameter = []
    num_iters = (num_epochs * Xs.shape[1]) // B
    for t in range(num_iters):
        if (t % monitor_period == 0):
            parameter.append(copy.deepcopy(W0))
        ii = []
        for b in range(B):
            ii.append(np.random.randint(0, Xs.shape[1]))
        W0 -= alpha * multinomial_logreg_batch_grad(Xs, Ys, gamma, W0, ii = ii)
    parameter.append(W0)
    return parameter

# ALGORITHM 2: run stochastic gradient descent with minibatching and sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters vectors, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    parameter = []
    for t in range(num_epochs):
        for i in range(Xs.shape[1] // B):
            if i % monitor_period == 0:
                parameter.append(copy.deepcopy(W0))
            ii = [(i * B + b) for b in range(B)]
            W0 -= alpha * multinomial_logreg_batch_grad(Xs, Ys, gamma, W0, ii = ii)
    parameter.append(W0)
    return parameter


# ALGORITHM 3: run stochastic gradient descent with minibatching and without-replacement sampling
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters vectors, one every "monitor_period" batches
def sgd_minibatch_random_reshuffling(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    parameter = []
    sigma = np.random.permutation(Xs.shape[1])
    for t in range(num_epochs):
        for i in range(Xs.shape[1] // B):
            if i % monitor_period == 0:
                parameter.append(copy.deepcopy(W0))
            ii = sigma[(i * B):(i * B + B)]
            W0 -= alpha * multinomial_logreg_batch_grad(Xs, Ys, gamma, W0, ii = ii)
    parameter.append(W0)
    return parameter


# A special selector for SGD just to make the testing code look better
def sgd_selector(sgd_type, Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    if sgd_type == 'sgd_minibatch':
        return sgd_minibatch(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period)
    if sgd_type == 'sgd_minibatch_sequential_scan':
        return sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period)
    if sgd_type == 'sgd_minibatch_random_reshuffling':
        return sgd_minibatch_random_reshuffling(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period)


if __name__ == "__main__":
    start_all = datetime.datetime.now()

    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()

    TEST_BREAKER = '*' * 20

    # Part 2: Starter GD Speed Test
    print(f"{TEST_BREAKER}\nPart 2\n")
    gamma = 0.0001
    alpha = 1.0
    numberIter = 10
    monitorFreq = 10
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    start = datetime.datetime.now()
    W_starter = gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, numberIter, monitorFreq, False)
    end = datetime.datetime.now()
    print(f"Time taken for starter GD is:  {end-start}")

    # Part 3: Faster GD Speed Test
    print(f"{TEST_BREAKER}\nPart 3\n")
    gamma = 0.0001
    alpha = 1.0
    numberIter = 10
    monitorFreq = 10
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    start = datetime.datetime.now()
    W_faster = gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, numberIter, monitorFreq)
    end = datetime.datetime.now()
    print(f"Time taken for faster GD is:  {end-start}")

    # Part 4: Evaluating GD
    # not finished
    print(f"{TEST_BREAKER}\nPart 4\n")
    gamma = 0.0001
    alpha = 1.0
    numberIter = 1000
    monitorFreq = 10
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    W_faster = gradient_descent(Xs_tr, Ys_tr, gamma, W, alpha, numberIter, monitorFreq)
    error_training = []
    error_testing = []
    loss_training = []
    loss_testing = []
    for w in W_faster:
        loss_training.append(multinomial_logreg_batch_loss(Xs_tr, Ys_tr, gamma, w))
        error_training.append(multinomial_logreg_error(Xs_tr, Ys_tr, w))
        loss_testing.append(multinomial_logreg_batch_loss(Xs_te, Ys_te, gamma, w))
        error_testing.append(multinomial_logreg_error(Xs_te, Ys_te, w))

    # Part 5.2: Evaluating SGD without minibatching
    # not finished
    print(f"{TEST_BREAKER}\nPart 5.2\n")
    sgd_list = ['sgd_minibatch', 'sgd_minibatch_sequential_scan', 'sgd_minibatch_random_reshuffling']
    gamma = 0.0001
    alpha = 0.001
    num_epochs = 10
    B = 1
    monitor_period = 6000
    model_list = {}
    for sgd_type in sgd_list:
        W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
        W_algo = sgd_selector(sgd_type, Xs_tr, Ys_tr, gamma, W, alpha, B, num_epochs, monitor_period)
        model_list[sgd_type] = copy.deepcopy(W_algo)
    


    # Part 5.3: Evaluating SGD with minibatching = 60
    # not finished
    print(f"{TEST_BREAKER}\nPart 5.3\n")
    gamma = 0.0001
    alpha = 0.05
    num_epochs = 10
    B = 60
    monitor_period = 100
    model_list1 = {}
    for sgd_type in sgd_list:
        W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
        W_algo = sgd_selector(sgd_type, Xs_tr, Ys_tr, gamma, W, alpha, B, num_epochs, monitor_period)
        model_list1[sgd_type] = copy.deepcopy(W_algo)
    
    #5.4; 5.5
    error_trmb = []
    error_temb = []
    error_trss = []
    error_tess = []
    error_trrs = []
    error_ters = []
    for w in model_list1['sgd_minibatch']:
        error_trmb.append(multinomial_logreg_error(Xs_tr, Ys_tr, w))
        error_temb.append(multinomial_logreg_error(Xs_te, Ys_te, w))
    for w in model_list1['sgd_minibatch_sequential_scan']:
        error_trss.append(multinomial_logreg_error(Xs_tr, Ys_tr, w))
        error_tess.append(multinomial_logreg_error(Xs_te, Ys_te, w))
    for w in model_list1['sgd_minibatch_random_reshuffling']:
        error_trrs.append(multinomial_logreg_error(Xs_tr, Ys_tr, w))
        error_ters.append(multinomial_logreg_error(Xs_te, Ys_te, w))
    error_tess = error_tess[1:][::10]
    error_trss = error_trss[1:][::10]
    error_trrs = error_trrs[1:][::10]
    error_ters = error_ters[1:][::10]
    error_trmb = error_trmb[1:][::10]
    error_temb = error_temb[1:][::10]

    error_training_mb = []
    error_testing_mb = []
    for w in model_list['sgd_minibatch']:
        error_training_mb.append(multinomial_logreg_error(Xs_tr, Ys_tr, w))
        error_testing_mb.append(multinomial_logreg_error(Xs_te, Ys_te, w))
    error_testing_mb_epoch = error_testing_mb[1:][::10]
    error_training_mb_epoch = error_training_mb[1:][::10]

    error_training_ss = []
    error_testing_ss = []
    for w in model_list['sgd_minibatch_sequential_scan']:
        error_training_ss.append(multinomial_logreg_error(Xs_tr, Ys_tr, w))
        error_testing_ss.append(multinomial_logreg_error(Xs_te, Ys_te, w))
    error_testing_ss_epoch = error_testing_ss[1:][::10]
    error_training_ss_epoch = error_training_ss[1:][::10]

    error_training_rs = []
    error_testing_rs = []
    for w in model_list['sgd_minibatch_random_reshuffling']:
        error_training_rs.append(multinomial_logreg_error(Xs_tr, Ys_tr, w))
        error_testing_rs.append(multinomial_logreg_error(Xs_te, Ys_te, w))
    error_testing_rs_epoch = error_testing_rs[1:][::10]
    error_training_rs_epoch = error_training_rs[1:][::10]

    from matplotlib import pyplot as plt
    x_values = [i for i in range(1, 11)]
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=150, figsize=(10, 10), sharex=False)  # 2 rows, 1 column

    # Plot the first subplot (ax1)
    ax1.plot(x_values, error_training_rs_epoch, color='red', label='rs')
    ax1.plot(x_values, error_training_mb_epoch, color = 'blue', label = 'mb')
    ax1.plot(x_values, error_training_ss_epoch, color = 'yellow', label = 'ss')
    ax1.plot(x_values, error_trmb, color = 'black', label = 'mb_60')
    ax1.plot(x_values, error_trss, color = 'pink', label = 'ss_60')
    ax1.plot(x_values, error_trrs, color = 'grey', label = 'rs_60')
    ax1.set_ylabel('Training Error')
    ax1.set_xlabel('Num of Epochs')
    ax1.set_title('Training')
    # Plot the second subplot (ax2)
    ax2.plot(x_values, error_testing_rs_epoch, color='red', label='rs')
    ax2.plot(x_values, error_testing_mb_epoch, color = 'blue', label = 'mb')
    ax2.plot(x_values, error_testing_ss_epoch, color = 'yellow', label = 'ss')
    ax2.plot(x_values, error_temb, color = 'black', label = 'mb_60')
    ax2.plot(x_values, error_tess, color = 'pink', label = 'ss_60')
    ax2.plot(x_values, error_ters, color = 'grey', label = 'rs_60')
    ax2.set_ylabel('Testing Error')
    ax2.set_xlabel('Num of Epochs')
    ax2.set_title('Testing')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    ax1.legend()
    ax2.legend()
    plt.show()

    # Part 6.1: step size (alpha) Tunning
    # not finished
    print(f"{TEST_BREAKER}\nPart 6.1\n")
    alpha_list = [0.0005, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
    
    num_epochs = 10
    gamma = 0.0001
    B = 1
    monitor_period = 6000
    
    best_alpha = None
    best_average_error = float('inf')
    
    
    tr, te = [[] for _ in range(len(alpha_list))], [[] for _ in range(len(alpha_list))]
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    model_list = {}
    for i in range(len(alpha_list)):
        alpha = alpha_list[i]

        W_algo_6_1 = sgd_selector('sgd_minibatch', Xs_tr, Ys_tr, gamma, W, alpha, B, num_epochs, monitor_period)
        model_list[i] = W_algo_6_1
        for j in W_algo_6_1:
            tr[i].append(multinomial_logreg_error(Xs_tr, Ys_tr, j))
            te[i].append(multinomial_logreg_error(Xs_te, Ys_te, j))

    # Part 6.2: step size (alpha) and minibatch size (B) Tunning
    # not finished
    print(f"{TEST_BREAKER}\nPart 6.2\n")
    B = 32
    alpha = 0.8
    gamma = 0.0001
    num_epochs = 10
    monitor_period = 9375
    result = []
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    W_algo_6_2 = sgd_selector('sgd_minibatch', Xs_tr, Ys_tr, gamma, W, alpha, B, num_epochs, monitor_period)
    train_error_epoch10 = multinomial_logreg_error(Xs_tr, Ys_tr, W_algo_6_2[-1])
    test_error_epoch10 = multinomial_logreg_error(Xs_te, Ys_te, W_algo_6_2[-1])
    print(f'when alpha is equal to {alpha} and batch size is equal to {B} then the training error is {train_error_epoch10} and test error is {test_error_epoch10}')
    
    # Part 6.3: step size (alpha) Tunning for less epoch
    # not finished
    print(f"{TEST_BREAKER}\nPart 6.3\n")
    #6.3
    B = 32
    alpha = 0.1
    gamma = 0.0001
    num_epochs = 10
    monitor_period = 4688
    result = []
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    W_algo_6_2 = sgd_selector('sgd_minibatch', Xs_tr, Ys_tr, gamma, W, alpha, B, num_epochs, monitor_period)
    train_error_epoch5 = multinomial_logreg_error(Xs_tr, Ys_tr, W_algo_6_2[-2])
    test_error_epoch5 = multinomial_logreg_error(Xs_te, Ys_te, W_algo_6_2[-2])
    print(f'when alpha is equal to {alpha} and batch size is equal to {B} then the training error is {train_error_epoch5} and test error is {test_error_epoch5}')
    #This value gives the training error for the new parameters at epoch 5 and clearly it is smaller


    # Part 6.4: plotting
    # not finished
    print(f"{TEST_BREAKER}\nPart 6.4\n")
    Bs = [32, 32, 64]
    alphas = [0.5, 0.1, 0.1]
    gamma = 0.0001
    num_epochs = 10
    monitor_periods = [int((10 * 60000 / Bs[i])/10)+1 for i in range(len(Bs))]
    tr, te = [[] for _ in range(3)], [[] for _ in range(3)]
    W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
    model_list = {}
    for i in range(len(Bs)):
        B = Bs[i]
        alpha = alphas[i]
        monitor_period = monitor_periods[i]
        W_algo_6_3 = sgd_selector('sgd_minibatch', Xs_tr, Ys_tr, gamma, W, alpha, B, num_epochs, monitor_period)
        model_list[i] = W_algo_6_3
        for j in W_algo_6_3:
            tr[i].append(multinomial_logreg_error(Xs_tr, Ys_tr, j))
            te[i].append(multinomial_logreg_error(Xs_te, Ys_te, j))
    tr = [i[1:] for i in tr]
    te = [i[1:] for i in te]

    x_values = [i for i in range(1, 11)]
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=150, figsize=(10, 10), sharex=False)  # 2 rows, 1 column

    # Plot the first subplot (ax1)
    ax1.plot(x_values, tr[0], color='red', label='32-0.5')
    ax1.plot(x_values, tr[1], color = 'blue', label = '32-0.1')
    ax1.plot(x_values, tr[2], color = 'yellow', label = '64-0.1')
    ax1.set_ylabel('Training Error')
    ax1.set_xlabel('Num of Epochs')
    ax1.set_title('Training')
    # Plot the second subplot (ax2)
    ax2.plot(x_values, te[0], color='red', label='32-0.5')
    ax2.plot(x_values, te[1], color = 'blue', label = '32-0.1')
    ax2.plot(x_values, te[2], color = 'yellow', label = '64-0.1')
    ax2.set_ylabel('Testing Error')
    ax2.set_xlabel('Num of Epochs')
    ax2.set_title('Testing')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    ax1.legend()
    ax2.legend()
    plt.show()
    # Part 7: System Evaluation
    print(f"{TEST_BREAKER}\nPart 7\n")
    sgd_list = ['sgd_minibatch', 'sgd_minibatch_sequential_scan', 'sgd_minibatch_random_reshuffling']

    # 5.2 config
    gamma = 0.0001
    alpha = 0.001
    num_epochs = 10
    B = 1
    monitor_period = 6000

    for sgd_type in sgd_list:
        start = datetime.datetime.now()
        for _ in range(5):
            W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
            W_algo1 = sgd_selector(sgd_type, Xs_tr, Ys_tr, gamma, W, alpha, B, num_epochs, monitor_period)
        end = datetime.datetime.now()
        print(f"Average time taken for {sgd_type} using 5.2 config is:  {(end-start)/5}")

    # 5.3 config
    gamma = 0.0001
    alpha = 0.05
    num_epochs = 10
    B = 60
    monitor_period = 100

    for sgd_type in sgd_list:
        start = datetime.datetime.now()
        for _ in range(5):
            W = np.zeros([Ys_tr.shape[0], Xs_tr.shape[0]])
            W_algo1 = sgd_selector(sgd_type, Xs_tr, Ys_tr, gamma, W, alpha, B, num_epochs, monitor_period)
        end = datetime.datetime.now()
        print(f"Average time taken for {sgd_type} using 5.3 config is:  {(end-start)/5}")

    end_all = datetime.datetime.now()
    print(f"{TEST_BREAKER}\nFinal\n")
    print(f"Total Time to run everything{(end_all-start_all)}")