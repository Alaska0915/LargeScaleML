
import numpy as np
import collections
# import random
# import pandas as pd
# import time

### a function to create a unique increasing ID
### note that this is just a quick-and-easy way to create a global order
### it's not the only way to do it
global_order_counter = 0
def get_next_order():
    global global_order_counter
    rv = global_order_counter
    global_order_counter = global_order_counter + 1
    return rv

### a helper function to convert constants into BackproppableArray objects
def to_ba(x):
    if isinstance(x, BackproppableArray):
        return x
    elif isinstance(x, np.ndarray):
        return BackproppableArray(x)
    elif isinstance(x, float):
        return BackproppableArray(np.array(x))
    elif isinstance(x, int):
        return BackproppableArray(np.array(float(x)))
    else:
        raise Exception("could not convert {} to BackproppableArray".format(x))

### a class for an array that can be "packpropped-through"
class BackproppableArray(object):
    # np_array     numpy array that stores the data for this object
    def __init__(self, np_array, dependencies=[]):
        super().__init__()
        self.data = np_array

        # grad holds the gradient, an array of the same shape as data
        # before backprop, grad is None
        # during backprop before grad_fn is called, grad holds the partially accumulated gradient
        # after backprop, grad holds the gradient of the loss (the thing we call backward on)
        #     with respect to this array
        # if you want to use the same array object to call backward twice, you need to re-initialize
        #     grad to zero first
        self.grad = None

        # an counter that increments monotonically over the course of the application
        # we know that arrays with higher order must depend only on arrays with lower order
        # we can use this to order the arrays for backpropagation
        self.order = get_next_order()

        # a list of other BackproppableArray objects on which this array directly depends
        # we'll use this later to decide which BackproppableArray objects need to participate in the backward pass
        self.dependencies = dependencies

    # represents me as a string
    def __repr__(self):
        return "({}, type={})".format(self.data, type(self).__name__)

    # returns a list containing this array and ALL the dependencies of this array, not just
    #    the direct dependencies listed in self.dependencies
    # that is, this list should include this array, the arrays in self.dependencies,
    #     plus all the arrays those arrays depend on, plus all the arrays THOSE arrays depend on, et cetera
    # the returned list must only include each dependency ONCE
    def all_dependencies(self):
        # TODO: (1.1) implement some sort of search to get all the dependencies
        visited, queue = set(), collections.deque([self])
        visited.add(self)
        while queue:
            element = queue.popleft()
            for i in element.dependencies:
                if i not in visited:
                    visited.add(i)
                    queue.append(i)
        return list(visited)

    # compute gradients of this array with respect to everything it depends on
    def backward(self):
        # can only take the gradient of a scalar
        assert(self.data.size == 1)

        # depth-first search to find all dependencies of this array
        all_my_dependencies = self.all_dependencies()

        orders = [i.order for i in all_my_dependencies]
        order_map = dict(zip(orders, all_my_dependencies))
        sorted_dependencies = list(dict(sorted(order_map.items(), key = lambda item: item[0] , reverse=True)).values())
        for i in sorted_dependencies:
            i.grad = np.zeros(shape = i.data.shape)
        self.grad = np.ones(shape = self.data.shape)
        for i in sorted_dependencies:
            i.grad_fn()
        
        # TODO: (1.2) implement the backward pass to compute the gradients
        #   this should do the following
        #   (1) sort the found dependencies so that the ones computed last go FIRST
        #   (2) initialize and zero out all the gradient accumulators (.grad) for all the dependencies
        #   (3) set the gradient accumulator of this array to 1, as an initial condition
        #           since the gradient of a number with respect to itself is 1
        #   (4) call the grad_fn function for all the dependencies in the sorted reverse order

    # function that is called to process a single step of backprop for this array
    # when called, it must be the case that self.grad contains the gradient of the loss (the
    #     thing we are differentating) with respect to this array
    # this function should update the .grad field of its dependencies
    #
    # this should just say "pass" for the parent class
    #
    # child classes override this
    def grad_fn(self):
        pass
        

    # operator overloading
    def __add__(self, other):
        return BA_Add(self, to_ba(other))
    def __sub__(self, other):
        return BA_Sub(self, to_ba(other))
    def __mul__(self, other):
        return BA_Mul(self, to_ba(other))
    def __truediv__(self, other):
        return BA_Div(self, to_ba(other))

    def __radd__(self, other):
        return BA_Add(to_ba(other), self)
    def __rsub__(self, other):
        return BA_Sub(to_ba(other), self)
    def __rmul__(self, other):
        return BA_Mul(to_ba(other), self)
    def __rtruediv__(self, other):
        return BA_Div(to_ba(other), self)

    # TODO (2.2) Add operator overloading for matrix multiplication
    def __matmul__(self, other):
        return BA_MatMul(self, to_ba(other))
    
    def sum(self, axis=None, keepdims=True):
        return BA_Sum(self, axis)

    def reshape(self, shape):
        return BA_Reshape(self, shape)

    def transpose(self, axes = None):
        if axes is None:
            axes = range(self.data.ndim)[::-1]
        return BA_Transpose(self, axes)

# TODO: implement any helper functions you'll need to backprop through vectors

    def boardcast(self, original_grad, target_grad):
        original_grad_shape = original_grad.shape
        target_grad_shape = target_grad.shape

        if original_grad_shape == target_grad_shape:
            return original_grad

        curr_shape = target_grad_shape
        while len(curr_shape) < len(original_grad_shape):
            curr_shape = (1,) + curr_shape
        indices = ()
        for axis_num, axis in enumerate(original_grad_shape):
            if curr_shape[axis_num] == 1 and axis != 1:
                indices += (axis_num, )

        return original_grad.sum(indices).reshape(target_grad_shape)
    
# a class for an array that's the result of an addition operation
class BA_Add(BackproppableArray):
    # x + y
    def __init__(self, x, y):
        super().__init__(x.data + y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (2.3) improve grad fn for Add
        self.x.grad += self.boardcast(self.grad, self.x.grad)
        self.y.grad += self.boardcast(self.grad, self.y.grad)

# a class for an array that's the result of a subtraction operation
class BA_Sub(BackproppableArray):
    # x - y
    def __init__(self, x, y):
        super().__init__(x.data - y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Sub
        self.x.grad += self.boardcast(self.grad, self.x.grad)
        self.y.grad -= self.boardcast(self.grad, self.y.grad)

# a class for an array that's the result of a multiplication operation
class BA_Mul(BackproppableArray):
    # x * y
    def __init__(self, x, y):
        super().__init__(x.data * y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (1.3, 2.3) implement grad fn for Mul
        self.x.grad += self.boardcast(self.grad * self.y.data, self.x.grad)
        self.y.grad += self.boardcast(self.x.data * self.grad, self.y.grad)

# a class for an array that's the result of a division operation
class BA_Div(BackproppableArray):
    # x / y
    def __init__(self, x, y):
        super().__init__(x.data / y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        self.x.grad += self.boardcast(self.grad / (self.y.data), self.x.grad)
        self.y.grad -= self.boardcast((self.grad * self.x.data) / ((self.y.data) * (self.y.data)), self.y.grad)


# a class for an array that's the result of a matrix multiplication operation
class BA_MatMul(BackproppableArray):
    # x @ y
    def __init__(self, x, y):
        # we only support multiplication of matrices, i.e. arrays with shape of length 2
        assert(len(x.data.shape) == 2)
        assert(len(y.data.shape) == 2)
        super().__init__(x.data @ y.data, [x,y])
        self.x = x
        self.y = y

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for MatMul
        self.x.grad += self.grad.dot(self.y.data.T)
        self.y.grad += self.x.data.T.dot(self.grad)


# a class for an array that's the result of an exponential operation
class BA_Exp(BackproppableArray):
    # exp(x)
    def __init__(self, x):
        super().__init__(np.exp(x.data), [x])
        self.x = x

    def grad_fn(self):
        # TODO: (1.3) implement grad fn for Exp
        grad_x = self.grad * self.data
        self.x.grad += grad_x

def exp(x):
    if isinstance(x, BackproppableArray):
        return BA_Exp(x)
    else:
        return np.exp(x)

# a class for an array that's the result of an logarithm operation
class BA_Log(BackproppableArray):
    # log(x)
    def __init__(self, x):
        super().__init__(np.log(x.data), [x])
        self.x = x

    def grad_fn(self):
        grad_x = self.grad / self.x.data
        self.x.grad += grad_x

def log(x):
    if isinstance(x, BackproppableArray):
        return BA_Log(x)
    else:
        return np.log(x)

# TODO: Add your own function
# END TODO

# a class for an array that's the result of a sum operation
class BA_Sum(BackproppableArray):
    # x.sum(axis, keepdims=True)
    def __init__(self, x, axis):
        super().__init__(x.data.sum(axis, keepdims=True), [x])
        self.x = x
        self.axis = axis

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for Sum
        self.x.grad += self.grad.sum(self.axis, keepdims=True)

# a class for an array that's the result of a reshape operation
class BA_Reshape(BackproppableArray):
    # x.reshape(shape)
    def __init__(self, x, shape):
        super().__init__(x.data.reshape(shape), [x])
        self.x = x
        self.shape = shape

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for Reshape
        self.x.grad += self.grad.reshape(self.x.data.shape)

# a class for an array that's the result of a transpose operation
class BA_Transpose(BackproppableArray):
    # x.transpose(axes)
    def __init__(self, x, axes):
        super().__init__(x.data.transpose(axes), [x])
        self.x = x
        self.axes = axes

    def grad_fn(self):
        # TODO: (2.1) implement grad fn for Transpose
        self.x.grad += self.grad.transpose(self.axes)


# numerical derivative of scalar function f at x, using tolerance eps
def numerical_diff(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps))/(2*eps)

def numerical_grad(f, x, eps=1e-5):
    # TODO: (2.5) implement numerical gradient function
    #       this should compute the gradient by applying something like
    #       numerical_diff independently for each entry of the input x
    num_diff = []
    for i in range(x.size):
        e = np.zeros(x.size)
        e[i] = 1
        e = e.reshape(x.shape)
        num_diff.append((f(x + eps*e) - f(x - eps*e))/(2*eps))
    return np.array(num_diff).reshape(x.shape)

# automatic derivative of scalar function f at x, using backprop
def backprop_diff(f, x):
    ba_x = to_ba(x)
    fx = f(ba_x)
    fx.backward()
    return ba_x.grad



# class to store test functions
class TestFxs(object):
    # scalar-to-scalar tests
    @staticmethod
    def f1(x):
        return x * 2 + 3

    @staticmethod
    def df1dx(x):
        # TODO (1.4) implement symbolic derivative of f1
        return 2

    @staticmethod
    def f2(x):
        return x * x

    @staticmethod
    def df2dx(x):
        # TODO (1.4) implement symbolic derivative of f2
        return 2*x

    @staticmethod
    def f3(x):
        u = (x - 2.0)
        return u / (u*u + 1.0)

    @staticmethod
    def df3dx(x):
        # TODO (1.4) implement symbolic derivative of f3
        u = (x-2.0)
        return (1 - u * u) / ((u*u + 1.0) ** 2)

    @staticmethod
    def f4(x):
        return log(exp(x*x / 8 - 3*x + 5) + x)

    # scalar-to-scalar tests that use vectors in the middle
    @staticmethod
    def g1(x):
        a = np.ones(3,dtype="float64")
        ax = x + a
        return (ax*ax).sum().reshape(())

    @staticmethod
    def g2(x):
        a = np.ones((4,5),dtype="float64")
        b = np.arange(20,dtype="float64")
        ax = x - a
        bx = log((x + b)*(x + b)).reshape((4,5)).transpose()
        y = bx @ ax
        return y.sum().reshape(())

    # vector-to-scalar tests
    @staticmethod
    def h1(x):  # takes an input of shape (5,)
        b = np.arange(5,dtype="float64")
        xb = x * b - 4
        return (xb * xb).sum().reshape(())

    # TODO: Add any other test functions you want to use here
    # END TODO
    @staticmethod
    def h2(x):#takes an input of shape (1000, )
        b = np.arange(1000,dtype="float64")
        xb = x * b - 4
        return (xb * xb).sum().reshape(())

    @staticmethod
    def h4(x):#takes an input of shape (1000, )
        return log(exp(x*x / 8 - 3*x + 5) + x).sum().reshape(())

    @staticmethod
    def h5(x):#takes an input of shape (1000, )
        a = np.ones(1000,dtype="float64")
        ax = x + a
        return (ax*ax).sum().reshape(())


if __name__ == "__main__":
    pass
    # TODO: Test your code using the provided test functions and your own functions
    #(1)get symbolic derivative implementation result
    # random.seed(42)
    # t1 = TestFxs()
    # random_numbers = [random.randint(1, 100) for _ in range(10)]
    # df1 = [t1.df1dx(i) for i in random_numbers]
    # df2 = [t1.df2dx(i) for i in random_numbers]
    # df3 = [t1.df3dx(i) for i in random_numbers]
    # #(2)get numerical derivative
    # df1_2 = [numerical_diff(t1.f1, i) for i in random_numbers]
    # df2_2 = [numerical_diff(t1.f2, i) for i in random_numbers]
    # df3_2 = [numerical_diff(t1.f3, i) for i in random_numbers]
    # df4_2 = [numerical_diff(t1.f4, i) for i in random_numbers]
    # #(3)get automatic differentiation
    # df1_3 = [backprop_diff(t1.f1, i) for i in random_numbers]
    # df2_3 = [backprop_diff(t1.f2, i) for i in random_numbers]
    # df3_3 = [backprop_diff(t1.f3, i) for i in random_numbers]
    # df4_3 = [backprop_diff(t1.f4, i) for i in random_numbers]
    # Test_case1 = pd.DataFrame({'symbolic':df1, 'numerical': df1_2, 'automatic': df1_3})
    # Test_case2 = pd.DataFrame({'symbolic':df2, 'numerical': df2_2, 'automatic': df2_3})
    # Test_case3 = pd.DataFrame({'symbolic':df3, 'numerical': df3_2, 'automatic': df3_3})
    # Test_case4 = pd.DataFrame({'numerical': df4_2, 'automatic': df4_3})
    # print(Test_case1)
    # print(Test_case2)
    # print(Test_case3)
    # print(Test_case4)
    # ########2(4)
    # df2_1_2 = [numerical_diff(t1.g1, i) for i in random_numbers]
    # df2_1_3 = [backprop_diff(t1.g1, i).sum() for i in random_numbers]
    # Test_case5 = pd.DataFrame({'numerical': df2_1_2, 'automatic': df2_1_3})
    # df2_2_2 = [numerical_diff(t1.g2, i) for i in random_numbers]
    # df2_2_3 = [backprop_diff(t1.g2, i).sum() for i in random_numbers]
    # Test_case6 = pd.DataFrame({'numerical': df2_2_2, 'automatic': df2_2_3})
    # print(Test_case6)
    # #########2(6)
    # random_vector = [np.random.rand(5, 1) for _ in range(10)]
    # df3_1_2 = [numerical_grad(t1.h1, i) for i in random_vector]
    # df3_1_3 = [backprop_diff(t1.h1, i) for i in random_vector]
    # Test_case7 = pd.DataFrame({'numerical': df3_1_2, 'automatic': df3_1_3})
    # print(Test_case7)
    # #########2(7)
    # random_vector1 = [np.random.rand(1000, 1) for _ in range(10)]
    # s1 = time.time()
    # df4_1_2 = [numerical_grad(t1.h2, i) for i in random_vector1]
    # e1 = time.time()
    # print(f'numerical grad takes {e1-s1}s')
    # s2 = time.time()
    # df4_1_3 = [backprop_diff(t1.h2, i) for i in random_vector1]
    # e2 = time.time()
    # print(f'backpropgation takes {e2-s2}s')
    # Test_case8 = pd.DataFrame({'numerical': df4_1_2, 'automatic': df4_1_3})
    # print(Test_case8)

    # #########h(3)
    # #s3 = time.time()
    # #df4_2_2 = [numerical_grad(t1.h3, i) for i in random_vector1]
    # #e3 = time.time()
    # #print(f'numerical grad takes {e3-s3}s')

    # #s4 = time.time()
    # #df4_2_3 = [backprop_diff(t1.h3, i) for i in random_vector1]
    # #e4 = time.time()
    # #print(f'backpropgation takes {e4 - s4}s')
    # #print(f'speed ratio is {((e3-s3)/(e4 - s4))*1000}ms')
    # #Test_case8 = pd.DataFrame({'numerical': df4_2_2, 'automatic': df4_2_3})
    # #print(Test_case8)

    # #########h(4)
    # s5 = time.time()
    # df4_3_2 = [numerical_grad(t1.h4, i) for i in random_vector1]
    # e5 = time.time()
    # print(f'numerical grad takes {e5 - s5}s')
    # s6 = time.time()
    # df4_3_3 = [backprop_diff(t1.h4, i) for i in random_vector1]
    # e6 = time.time()
    # print(f'backpropgation takes {e6 - s6}s')
    # print(f'speed ratio is {(e5-s5)/(e6 - s6)}s')
    # Test_case9 = pd.DataFrame({'numerical': df4_3_2, 'automatic': df4_3_3})
    # print(Test_case9)

    # #########h(5)
    # s7 = time.time()
    # df4_4_2 = [numerical_grad(t1.h5, i) for i in random_vector1]
    # e7 = time.time()
    # print(f'numerical grad takes {e7 - s7}s')
    # s8 = time.time()
    # df4_4_3 = [backprop_diff(t1.h5, i) for i in random_vector1]
    # e8 = time.time()
    # print(f'backpropgation takes {e8 - s8}s')
    # print(f'speed ratio is {(e7-s7)/(e8 - s8)}s')
    # Test_case10 = pd.DataFrame({'numerical': df4_4_2, 'automatic': df4_4_3})
    # print(Test_case10)
    # n_speed = [e1- s1, e5 - s5, e7 - s7]
    # a_speed = [e2 - s2, e6 - s6, e8 - s8]
    # ratios = [n_speed[i]/a_speed[i] for i in range(len(n_speed))]
    # Speed_summary = pd.DataFrame({'numerical(s)':n_speed, 'Backpropgation(s)':a_speed, 'Ratio':ratios})
    # Test_case1.to_csv('T1.csv', index=False)
    # Test_case2.to_csv('T2.csv', index=False)
    # Test_case3.to_csv('T3.csv', index=False)
    # Test_case4.to_csv('T4.csv', index=False)
    # Test_case5.to_csv('T5.csv', index=False)
    # Test_case6.to_csv('T6.csv', index=False)
    # Test_case7.to_csv('T7.csv', index=False)
    # Test_case8.to_csv('T8.csv', index=False)
    # Test_case9.to_csv('T9.csv', index=False)
    # Test_case10.to_csv('T10.csv', index=False)
    # Speed_summary.to_csv('Summary.csv', index = False)








    




    
