import pandas as pd

dtypes = {'col1': 'float', 'col2': 'float', 'col3': 'int'}

df = pd.read_csv('/Users/goncalo/Documents/University/OML/Class 1/data1.csv',  sep = ' ', header = None, dtype = dtypes)

N = df.shape[0]
x = df.iloc[:, [0, 1]].values.tolist()
y = df.iloc[:, 2].values.tolist()


# Signal of the cross product between two lists
def sgn(list1, list2):
    product = 0 
    for i in range(0, len(list1)): 
        product = product + (list1[i] * list2[i])
    if product > 0: return 1
    else: return -1


# Multiply each element of a list with a value
def multList(value, list1):
    result = []
    for i in range(0, len(list1)):
        result.append(list1[i] * value)
    return result


# Element-wise addition of two lists
def addLists(list1, list2):
    result = []
    for i in range(0, len(list1)):
        result.append(list1[i] + list2[i])
    return result


# Cost function
def costFun(w):
    result = 0
    for n in range(0, N):
        x_tilde = [1.0] + x[n]
        y_hat = sgn(w, x_tilde)
        result = result + (1/2 * abs(y[n] - y_hat))
    return result / N


# Show perceptron stats
def showStats(t, w, error):
    print('t = %s' % t)
    print('w(%s) = %s' % (t, w))
    print('E(%s) = %s' % (t, error))
    print('---')


# Perceptron algorithm v1 (primal)
def perceptron():
    w = [-0.25] * (N - 1)
    t = 0
    error = costFun(w)
    showStats(t, w, error)
    while True:
        for n in range(0, N):
            x_tilde = [1.0] + x[n]
            y_hat = sgn(w, x_tilde)
            if (y[n] != y_hat):
                w = addLists(w, multList(y[n], x_tilde))
                error = costFun(w)
                if error == 0:
                    t = t + 1
                    showStats(t, w, error)
                    return 1
            t = t + 1
            showStats(t, w, error)


perceptron()