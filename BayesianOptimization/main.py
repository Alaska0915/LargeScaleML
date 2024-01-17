#!/usr/bin/env python3
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import matplotlib
import pickle
import numpy
import scipy.special
import mnist
from tqdm import tqdm
matplotlib.use('agg')
from matplotlib import pyplot
pyplot.rcParams['animation.ffmpeg_path'] = 'C:/Users/15109/Downloads/ffmpeg-6.1/ffmpeg-6.1'
from matplotlib import animation
import torch
## you may wish to import other things like torch.nn
import warnings
warnings.filterwarnings("ignore")
import time

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

### hyperparameter settings and other constants
TEST_BREAKER = '*' * 20
gamma, sigma2_noise, kappa = 10, 0.001, 2
gd_alpha, gd_nruns, gd_niters = 0.01, 20, 20
n_warmup, num_iters = 3, 20

B, num_epochs = 500, 5
### end hyperparameter settings

def load_MNIST_dataset_with_validation_split():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = numpy.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        numpy.random.seed(8675309)
        perm = numpy.random.permutation(60000)
        Xs_tr = numpy.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = numpy.ascontiguousarray(Ys_tr[:,perm])
        # extract out a validation set
        Xs_va = Xs_tr[:,50000:60000]
        Ys_va = Ys_tr[:,50000:60000]
        Xs_tr = Xs_tr[:,0:50000]
        Ys_tr = Ys_tr[:,0:50000]
        # load test data
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = numpy.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = numpy.ascontiguousarray(Xs_te)
        Ys_te = numpy.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the cumulative distribution function of a standard Gaussian random variable
def gaussian_cdf(u):
    return 0.5*(1.0 + torch.special.erf(u/math.sqrt(2.0)))

# compute the probability mass function of a standard Gaussian random variable
def gaussian_pmf(u):
    return torch.exp(-u**2/2.0)/math.sqrt(2.0*math.pi)


# compute the Gaussian RBF kernel matrix for a vector of data points (in PyTorch)
#
# Xs        points at which to compute the kernel (size: d x m)
# Zs        other points at which to compute the kernel (size: d x n)
# gamma     gamma parameter for the RBF kernel
#
# returns   an (m x n) matrix Sigma where Sigma[i,j] = K(Xs[:,i], Zs[:,j])
def rbf_kernel_matrix(Xs, Zs, gamma):
    # TODO students should implement this
    m, n = Xs.size(1), Zs.size(1)
    inner = -2 * torch.matmul(Xs.T, Zs)
    dist = torch.reshape(torch.sum(Xs ** 2, axis=0), [m, 1]) + torch.reshape(torch.sum(Zs ** 2, axis=0), [1, n])
    return torch.exp(-gamma * (dist + inner))

# compute the distribution predicted by a Gaussian process that uses an RBF kernel (in PyTorch)
#
# Xs            points at which to compute the kernel (size: d x n) where d is the number of parameters
# Ys            observed value at those points (size: n)
# gamma         gamma parameter for the RBF kernel
# sigma2_noise  the variance sigma^2 of the additive gaussian noise used in the model
#
# returns   a function that takes a value Xtest (size: d) and returns a tuple (mean, variance)
def gp_prediction(Xs, Ys, gamma, sigma2_noise):
    # first, do any work that can be shared among predictions
    # TODO students should implement this
    sigma = rbf_kernel_matrix(Xs, Xs, gamma) + sigma2_noise * torch.eye(Xs.size(1))
    toMat = lambda x: x.view(-1, 1)
    inv = torch.inverse(sigma)
    inv_y = torch.matmul(inv, toMat(Ys))
    # next, define a nested function to return
    def prediction_mean_and_variance(Xtest):
        # TODO students should implement this
        # construct mean and variance
        k_vec = lambda x: rbf_kernel_matrix(Xs, toMat(x), gamma)
        mean = torch.matmul(k_vec(Xtest).T, inv_y)
        quad = lambda x: torch.matmul(k_vec(x).T, torch.matmul(inv, k_vec(x)))
        variance = rbf_kernel_matrix(toMat(Xtest), toMat(Xtest), gamma) + sigma2_noise - quad(Xtest)
        return (mean.reshape(()), variance.reshape(())) # be sure to return scalars!
    #finally, return the nested function
    return prediction_mean_and_variance


# compute the probability of improvement (PI) acquisition function
#
# Ybest     value at best "y"
# mean      mean of prediction
# stdev     standard deviation of prediction (the square root of the variance)
#
# returns   PI acquisition function
def pi_acquisition(Ybest, mean, stdev):
    # TODO students should implement this
    return - gaussian_cdf((Ybest - mean) / stdev)


# compute the expected improvement (EI) acquisition function
#
# Ybest     value at best "y"
# mean      mean of prediction
# stdev     standard deviation of prediction
#
# returns   EI acquisition function
def ei_acquisition(Ybest, mean, stdev):
    # TODO students should implement this
    Z = (Ybest - mean) / stdev
    return - (gaussian_pmf(Z) + Z * gaussian_cdf(Z) ) * stdev


# return a function that computes the lower confidence bound (LCB) acquisition function
#
# kappa     parameter for LCB
#
# returns   function that computes the LCB acquisition function
def lcb_acquisition(kappa):
    def A_lcb(Ybest, mean, stdev):
        # TODO students should implement this
        return mean - kappa * stdev
    return A_lcb


def acquisition_helper(y_best, prediction_mean_and_variance, acquisition):
    def acq(x):
        (mean,variance) = prediction_mean_and_variance(x)
        std_dev = torch.math.sqrt(torch.clamp(variance, min = 0))
        temp = acquisition(y_best, mean, std_dev)
        return temp
    return acq


# gradient descent to do the inner optimization step of Bayesian optimization
#
# objective     the objective function to minimize, as a function that takes a torch tensor and returns an expression
# x0            initial value to assign to variable (torch tensor)
# alpha         learning rate/step size
# num_iters     number of iterations of gradient descent
#
# returns     (obj_min, x_min), where
#       obj_min     the value of the objective after running iterations of gradient descent
#       x_min       the value of x after running iterations of gradient descent
def gradient_descent(objective, x0, alpha, num_iters):
    x = x0.detach().clone()  # create a fresh copy of x0
    x.requires_grad = True   # make it a target for differentiation
    opt = torch.optim.SGD([x], alpha)
    for it in range(num_iters):
        opt.zero_grad()
        f = objective(x)
        f.backward()
        opt.step()
    x.requires_grad = False  # make x no longer require gradients
    return (float(f.item()), x)

# run Bayesian optimization to minimize an objective
#
# objective     objective function; takes a torch tensor, returns a python float scalar
# d             dimension to optimize over
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# acquisition   acquisition function to use (e.g. ei_acquisition)
# random_x      function that returns a random sample of the parameter we're optimizing over (a torch tensor, e.g. for use in warmup)
# gd_nruns      number of random initializations we should use for gradient descent for the inner optimization step
# gd_alpha      learning rate for gradient descent
# gd_niters     number of iterations for gradient descent
# n_warmup      number of initial warmup evaluations of the objective to use
# num_iters     number of outer iterations of Bayes optimization to run (including warmup)
#
# returns       tuple of (y_best, x_best, Ys, Xs), where
#   y_best          objective value of best point found
#   x_best          best point found
#   Ys              vector of objective values for all points searched (size: num_iters)
#   Xs              matrix of all points searched (size: d x num_iters)
def bayes_opt(objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters):
    # TODO students should implement this
    y_best = float('inf')
    x_best = random_x()
    Xs, Ys = torch.zeros((d, num_iters)), torch.zeros((num_iters))
    for iter in tqdm(range(n_warmup), desc='Warmup'):
        x = random_x()
        Xs[:, iter] = torch.tensor(x)
        y = objective(x)
        Ys[iter] = y
        if y <= y_best:
            x_best, y_best = x, y

    for iter in tqdm(range(n_warmup, num_iters), desc='BayesOpt'):
        prediction_mean_and_variance = gp_prediction(Xs, Ys, gamma, sigma2_noise)
        acq = acquisition_helper(y_best, prediction_mean_and_variance, acquisition)
        x_inner, y_inner = [], []
        for _ in range(gd_nruns):
            x = random_x()
            (obj_min, x_min) = gradient_descent(acq, x, gd_alpha, gd_niters)
            x_inner.append(x_min)
            y_inner.append(obj_min)

        ind = numpy.argmin(y_inner)
        x = x_inner[ind]
        y = objective(torch.tensor(x))
        Xs[:, iter], Ys[iter] = torch.tensor(x), y

        if y <= y_best:
            x_best, y_best = x, y

    if isinstance(y_best, torch.Tensor):
        y_best = float(y_best.item())
    
    return (y_best, x_best, Ys, Xs)

def bayes_opt_timed(objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters):
    # TODO students should implement this
    y_best = float('inf')
    x_best = random_x()
    Xs, Ys = torch.zeros((d, num_iters)), torch.zeros((num_iters))
    total_obj_time = 0
    time_start = time.time()
    for iter in tqdm(range(n_warmup), desc='Warmup'):
        x = random_x()
        Xs[:, iter] = torch.tensor(x)
        time1_start = time.time()
        y = objective(x)
        time1_end = time.time()
        total_obj_time += time1_end - time1_start
        Ys[iter] = y
        if y <= y_best:
            x_best, y_best = x, y

    for iter in tqdm(range(n_warmup, num_iters), desc='BayesOpt'):
        prediction_mean_and_variance = gp_prediction(Xs, Ys, gamma, sigma2_noise)
        acq = acquisition_helper(y_best, prediction_mean_and_variance, acquisition)
        x_inner, y_inner = [], []
        for _ in range(gd_nruns):
            x = random_x()
            (obj_min, x_min) = gradient_descent(acq, x, gd_alpha, gd_niters)
            x_inner.append(x_min)
            y_inner.append(obj_min)

        ind = numpy.argmin(y_inner)
        x = x_inner[ind]
        time2_start = time.time()
        y = objective(torch.tensor(x))
        time2_end = time.time()
        total_obj_time += time2_end - time2_start
        Xs[:, iter], Ys[iter] = torch.tensor(x), y

        if y <= y_best:
            x_best, y_best = x, y

    if isinstance(y_best, torch.Tensor):
        y_best = float(y_best.item())
    
    time_end = time.time()
    total_program_time = time_end - time_start
    fraction = total_obj_time / total_program_time
    return (y_best, x_best, Ys, Xs, fraction)

# a one-dimensional test objective function on which to run Bayesian optimization
def test_objective(x):
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1,)
    x = x.item() # convert to a python float
    return (math.cos(8.0*x) - 0.3 + (x-0.5)**2)



# compute the gradient of the multinomial logistic regression objective, with regularization (SIMILAR TO PROGRAMMING ASSIGNMENT 2)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_batch_grad(Xs, Ys, ii, gamma, W):
    # here is the code from my solution
    # you can also use your implementation from programming assignment 2

    #return numpy.dot(scipy.special.softmax(numpy.dot(W, Xs[:,ii]), axis=0) - Ys[:,ii], Xs[:,ii].transpose()) / len(ii) + gamma * W
    Xs = Xs[:,ii]
    Ys = Ys[:,ii]
    (d, n) = Xs.shape
    y_hat = scipy.special.softmax(numpy.dot(W,Xs), axis=0)
    del_L = numpy.dot(y_hat - Ys, Xs.T)
    grad = del_L / n + (gamma * W).numpy()
    return grad


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # here is the code from my solution
    # you can also use your implementation from programming assignment 2
    predictions = numpy.argmax(numpy.dot(W, Xs), axis=0)
    error = numpy.mean(predictions != numpy.argmax(Ys, axis=0))
    return error


# compute the cross-entropy loss of the classifier (SAME AS PROGRAMMING ASSIGNMENT 3)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    # here is the code from my solution
    # you can also use your implementation from programming assignment 3
    (d, n) = Xs.shape
    return -numpy.sum(numpy.log(scipy.special.softmax(numpy.dot(W, Xs), axis=0)) * Ys) / n + (gamma / 2) * (numpy.linalg.norm(W, "fro")**2)


# SGD + Momentum: add momentum to the previous algorithm
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
#
# returns         the final model, after training
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs):
    # here is the code from my solution
    # you may also adapt your implementation from PA3 if you prefer
    (d, n) = Xs.shape
    V = numpy.zeros(W0.shape)
    W = W0
    niter = 0
    for it in range(num_epochs):
        for ibatch in range(int(n/B)):
            niter += 1
            ii = range(ibatch*B, (ibatch+1)*B)
            V = beta * V - alpha * multinomial_logreg_batch_grad(Xs, Ys, ii, gamma, W)
            W = W + V.numpy()
    return W


# produce a function that runs SGD+Momentum on the MNIST dataset, initializing the weights to zero
#
# mnist_dataset         the MNIST dataset, as returned by load_MNIST_dataset_with_validation_split
# num_epochs            number of epochs to run for
# B                     the batch size
#
# returns               a function that takes parameters
#   params                  a numpy vector of shape (3,) with entries that determine the hyperparameters, where
#       gamma = 10^(-8 * params[0])
#       alpha = 0.5*params[1]
#       beta = params[2]
#                       and returns (the validation error of the final trained model after all the epochs) minus 0.9.
#                       if training diverged (i.e. any of the weights are non-finite) then return 0.1, which corresponds to an error of 1.
def mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B):
    # TODO students should implement this
    Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te = mnist_dataset
    c, _ = Ys_tr.shape
    d, _ = Xs_tr.shape
    W0 = numpy.zeros((c, d))

    def train(params):
        model = sgd_mss_with_momentum(Xs_tr, Ys_tr, 10**(-8 * params[0]), W0, 0.5 * params[1], params[2], B, num_epochs)
        if numpy.isinf(model).any() or numpy.isnan(model).any():
            return 0.1
        else:
            return multinomial_logreg_error(Xs_va, Ys_va, model) - 0.9
    return train


# produce an animation of the predictions made by the Gaussian process in the course of 1-d Bayesian optimization
#
# objective     objective function
# acq           acquisition function
# gamma         gamma to use for RBF hyper-hyperparameter
# sigma2_noise  additive Gaussian noise parameter for Gaussian Process
# Ys            vector of objective values for all points searched (size: num_iters)
# Xs            matrix of all points searched (size: d x num_iters)
# xs_eval       torch vector of xs at which to evaluate the mean and variance of the prediction at each step of the algorithm
# filename      path at which to store .mp4 output file
def animate_predictions(objective, acq, gamma, sigma2_noise, Ys, Xs, xs_eval, filename):
    mean_eval = []
    variance_eval = []
    acq_eval = []
    acq_Xnext = []
    for it in range(len(Ys)):
        print("rendering frame %i" % it)
        Xsi = Xs[:, 0:(it+1)]
        Ysi = Ys[0:(it+1)]
        ybest = Ysi.min()
        gp_pred = gp_prediction(Xsi, Ysi, gamma, sigma2_noise)
        pred_means = []
        pred_variances = []
        pred_acqs = []
        for x_eval in xs_eval:
            XE = x_eval.reshape(1)
            (pred_mean, pred_variance) = gp_pred(XE)
            pred_means.append(float(pred_mean))
            pred_variances.append(float(pred_variance))
            pred_acqs.append(float(acq(ybest, pred_mean, math.sqrt(pred_variance))))
        mean_eval.append(torch.Tensor(pred_means))
        variance_eval.append(torch.Tensor(pred_variances))
        acq_eval.append(torch.Tensor(pred_acqs))
        if it + 1 != len(Ys):
            XE = Xs[0,it+1].reshape(1)
            (pred_mean, pred_variance) = gp_pred(XE)
            acq_Xnext.append(float(acq(ybest, pred_mean, math.sqrt(pred_variance))))

    fig = pyplot.figure()
    fig.tight_layout()
    ax = fig.gca()
    ax2 = ax.twinx()

    def animate(i):
        ax.clear()
        ax2.clear()
        ax.set_xlabel("parameter")
        ax.set_ylabel("objective")
        ax2.set_ylabel("acquisiton fxn")
        ax.set_title("Bayes Opt After %d Steps" % (i+1))
        l1 = ax.fill_between(xs_eval, mean_eval[i] + 2.0*torch.sqrt(variance_eval[i]), mean_eval[i] - 2.0*torch.sqrt(variance_eval[i]), color="#eaf1f7")
        l2, = ax.plot(xs_eval, [objective(x.reshape(1)) for x in xs_eval])
        l3, = ax.plot(xs_eval, mean_eval[i], color="r")
        l4 = ax.scatter(Xs[0,0:(i+1)], Ys[0:(i+1)])
        l5, = ax2.plot(xs_eval, acq_eval[i], color="g", ls=":")
        ax.legend([l2, l3, l5], ["objective", "mean", "acquisition"], loc="upper right")
        if i + 1 == len(Ys):
            return l1, l2, l3, l4, l5
        else:
            l6 = ax2.scatter([Xs[0,i+1]], [acq_Xnext[i]], color="g")
            return l1, l2, l3, l4, l5, l6


    ani = animation.FuncAnimation(fig, animate, frames=range(len(Ys)), interval=600, repeat_delay=1000)

    ani.save(filename)

### test function setting

def part2_1():
    print(f"\n{TEST_BREAKER}\nPart 2.1: Simple Test Objective\n")
    d = 1
    random_x = lambda: torch.rand(d)
    optima = {}
    acquis = [pi_acquisition, ei_acquisition, lcb_acquisition(kappa)]
    names = ['pi', 'ei', 'lcb']
    for name, acquisition in zip(names,acquis):
        y_best, x_best, Ys, Xs = bayes_opt(test_objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
        optima[name] = (x_best, y_best)
    print(optima)
    return optima

def part2_2():
    print(f"\n{TEST_BREAKER}\nPart 2.2: Visualize\n")

    def test_random_x():
        return 1.5 * torch.rand(1) - 0.25

    (y_best, x_best, Ys, Xs) = bayes_opt(test_objective, 1, 10.0, 0.001, ei_acquisition, test_random_x, 20, 0.01, 20, 3, 20)
    print(y_best)
    print(x_best)
    print(Ys)
    print(Xs)
    Xs_plot = torch.linspace(-0.5, 1.5, steps=256)
    animate_predictions(test_objective, ei_acquisition, 10.0, 0.001, Ys, Xs, Xs_plot, "C:/Users/15109/Downloads/pa4/release/test.mp4")

def part2_3():
    print(f"\n{TEST_BREAKER}\nPart 2.3: Gamma Tunning\n")
    gammas = numpy.array([1e-6, 1e-3, 1, 1e3, 1e6]) * 10
    d = 1
    random_x = lambda: torch.rand(d)

    optima = {}
    acquis = [pi_acquisition, ei_acquisition, lcb_acquisition(kappa)]
    names = ['pi', 'ei', 'lcb']
    for name, acquisition in zip(names,acquis):
        for gamma in gammas:
            y_best, x_best, Ys, Xs = bayes_opt(test_objective, d, gamma, sigma2_noise, acquisition, random_x, gd_nruns,
                                               gd_alpha, gd_niters, n_warmup, num_iters)
            optima[(name,gamma)] = (y_best, x_best)
    print('The result from Gamma Tunning ' + str(optima))

def part2_4():
    print(f"\n{TEST_BREAKER}\nPart 2.4: Kappa Tunning\n")
    kappas = numpy.array([1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]) * 2
    d = 1
    random_x = lambda: torch.rand(d)

    optima = {}
    for kappa in kappas:
        y_best, x_best, Ys, Xs = bayes_opt(test_objective, d, gamma, sigma2_noise, lcb_acquisition(kappa), random_x, gd_nruns,
                                               gd_alpha, gd_niters, n_warmup, num_iters)
        optima[(kappa)] = (y_best, x_best)
    print('The result from Kappa Tunning ' + str(optima))

def part3_3():
    print(f"\n{TEST_BREAKER}\nPart 3.3: MNIST\n")

    d = 3
    random_x = lambda: torch.rand(3)
    mnist_dataset = load_MNIST_dataset_with_validation_split()
    objective = mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B)
    y_best, x_best, Ys, Xs = bayes_opt(objective, d, gamma, sigma2_noise, lcb_acquisition(kappa), random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)

    Xs_tr, Ys_tr, Xs_va, Ys_va, Xs_te, Ys_te = mnist_dataset
    c, _ = Ys_tr.shape
    d, _ = Xs_tr.shape
    W0 = numpy.zeros((c, d))

    model = sgd_mss_with_momentum(Xs_tr, Ys_tr, 10**(-8 * x_best[0]), W0, 0.5 * x_best[1], x_best[2], B, num_epochs)
    va_error = multinomial_logreg_error(Xs_va, Ys_va, model) - 0.9
    te_error = multinomial_logreg_error(Xs_te, Ys_te, model) - 0.9
    best_gamma, best_alpha, best_beta = 10**(-8 * x_best[0]), 0.5 * x_best[1], x_best[2]
    print('The best parameters are' + str(x_best))
    print('Validation error %f, Test error %f' % (va_error, te_error))
    print(f'best gamma: {best_gamma}, best alpha: {best_alpha}, best beta: {best_beta}')


def part3_4():
    print(f"\n{TEST_BREAKER}\nPart 3.4: Wall Clock Analysis\n")

    #experiment 1 just follow what we did in 3.3 and run it only once
    d = 3
    random_x = lambda: torch.rand(3)
    mnist_dataset = load_MNIST_dataset_with_validation_split()
    objective = mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B)
    y_best, x_best, Ys, Xs, fraction = bayes_opt_timed(objective, d, gamma, sigma2_noise, lcb_acquisition(kappa), random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
    print(f'the fraction of time used on computing obj is {fraction}')

    #experiment 2 run it for 10 times with random sampling and record the fraction each time and get the average
    total_fraction = 0
    for _ in range(10):
        mnist_dataset = load_MNIST_dataset_with_validation_split()
        objective = mnist_sgd_mss_with_momentum(mnist_dataset, num_epochs, B)
        y_best, x_best, Ys, Xs, fraction = bayes_opt_timed(objective, d, gamma, sigma2_noise, lcb_acquisition(kappa), random_x, gd_nruns, gd_alpha, gd_niters, n_warmup, num_iters)
        total_fraction += fraction
    print(f'the fraction of time used on computing obj is {total_fraction/10}')


if __name__ == "__main__":
    part2_1()
    part2_2()
    part2_3()
    part2_4()
    part3_3()
    part3_4()


