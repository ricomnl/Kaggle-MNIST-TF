import numpy as np
from modeltf import forward_propagation

def predict(X, parameters):
    Yhat, _ = forward_propagation(X, parameters)
    Yhat = np.argmax(Yhat, axis = 0)
    
    return Yhat

def get_error(Yhat, Y):
    if Y.shape[0] != 1:
        Y = np.argmax(Y, axis = 1)
    error = np.sum(Yhat == Y)/Yhat.shape[0]
    
    return error
    