import numpy as np
class LAMBDA():# state-based parametric lambda
    def __init__(self, env, initial_value, approximator = 'constant'):
        self.n = env.observation_space.n
        self.approximator = approximator
        if approximator == 'constant':
            self.w = initial_value.reshape(-1)
        elif approximator == 'linear':
            self.w = initial_value.reshape(-1)
        elif approximator == 'NN':
            pass # Neural Network approximator to be implemented using PyTorch

    def value(self, x):
        if self.approximator == 'constant':
            l = self.w
        elif self.approximator == 'linear':
            l = np.dot(x.reshape(-1), self.w)
        elif self.approximator == 'NN':
            pass # not implemented
        if l > 1:
            print('lambda value greater than 1, truncated to 1')
            return 1
        elif l < 0:
            print('lambda value less than 0, truncated to 0')
            return 0
        return l

    def gradient(self, x):
        if self.approximator == 'linear':
            return x.reshape(-1)

    def gradient_descent(self, x, step_length):
        gradient = self.gradient(x)
        value_after = np.dot(x.reshape(-1), (self.w - step_length * gradient))
        if value_after > 1:
            pass # overflow of lambda rejected
        elif value_after < 0:
            pass # underflow of lambda rejected
        else:
            self.w -= step_length * gradient