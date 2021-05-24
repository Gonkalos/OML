import numpy as np
import csv
from random import randint

from numpy.core.fromnumeric import diagonal

'''
Load data from file
'''
def load_data(filename):    
    file = open(filename, 'r') 
    tmp_str = file.readline()
    tmp_arr = tmp_str[:-1].split(' ')
    N = int(tmp_arr[0])
    n_row = int(tmp_arr[1])
    n_col = int(tmp_arr[2])
    print('N=%d, row=%d, col=%d' %(N,n_row,n_col))
    data = np.zeros([N, n_row * n_col + 1])
    for n in range(N):
        tmp_str = file.readline()
        tmp_arr = tmp_str[:-1].split(' ')       
        for i in range(n_row * n_col + 1):
            data[n][i] = int(tmp_arr[i])
    file.close() 
    return N, n_row, n_col, data

'''
Sigmoid function
'''
def sigmoid(s):  
    large = 30
    if s < -large: 
        s = -large
    if s > large: 
        s = large
    return (1 / (1 + np.exp(-s)))

'''
Cost funtion
'''
def cost(X, Y, N, ew, b, ev, c):
    sum = 0
    for n in range(N):
        prediction = predict(X[n], ew, b)
        sum += Y[n] * np.log(prediction) + (1 - Y[n]) * np.log(1 - prediction)
    E = - sum / N
    return E

'''
Predict label
'''
def predict(x, ew, b):
    w = ew[1:]
    s = b + np.dot(x, w)
    sigma = sigmoid(s)
    return sigma

'''
'''
def update(X, Y, I, m, eta, eh, ew, b, ev, c):
    prediction = predict(X[m], ew, b)
    # auxiliar matrices
    h = eh[1:]
    w = ew[1:]
    v = ev[1:]
    func1 = lambda a: a * (1 - a)
    diagonal = np.array([func1(h_m) for h_m in h])
    matrix1 = np.diag(diagonal) # J x J
    matrix2 = np.tile(X, (I, 1))        
    matrix2 = matrix2 * v # J x I
    # update w
    grad_w = (prediction - Y[m]) * np.matmul(matrix1, matrix2) # w's gradient
    w = w - eta * grad_w                                       # w (t + 1)
    # update b
    grad_b = (prediction - Y[m]) * (matrix1 * v) # b's gradient
    b = b - eta * grad_b                         # b (t + 1)
    # update v
    grad_v = (prediction - Y[m]) * h # v's gradient
    v = v - eta * grad_v             # v (t + 1)
    # update c
    grad_c = prediction - Y[m] # c's gradient
    c = c - eta * grad_c       # c (t + 1)
    return w, b, v, c

'''
Run shallow logistic classifier
'''
def run_slc(X, Y, N, I, eta, max_iteration, eh, ew, b, ev, c, errors):
    epsi = 10e-3
    epoch = 0
    while (errors[-1] > epsi):
        epoch += 1
        if (epoch > max_iteration):
            break
        # choose random data from dataset
        m = randint(0, N - 1)
        # update w, b, v, c from (t) to (t + 1)
        w, b, v, c = update(X, Y, I, m, eta, eh, ew, b, ev, c)
        error = cost(X, Y, N, ew, b, ev, c)
        errors.append(error)
    return w, b, v, c, errors
        
'''
Main execution
'''

# load data from dataset
N, n_row, n_col,data = load_data('./Data/XOR.txt')
N = int(N * 1.0)
I = n_row * n_col
X = data[:N, :-1] 
Y = data[:N, -1]

J = 3 # number of neurons in the hidden layer

# initialize w, b, v, c
ew = np.ones([I + 1]) # w + bias
b = 1
ev = np.ones([J + 1]) # v + bias
c = 1
eh = np.ones([J + 1]) # hidden layer's weigths

eta = 0.1 # learning rate

# initialize errors' list
errors = []
errors.append(cost(X, Y, N, ew, b, ev, c))