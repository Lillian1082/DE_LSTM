import theano
import theano.tensor as T
import numpy as np

epsilon = 1.0e-9
# C-delata function
def C_delta(delta_y, min_delta, interval_delta, num_class):
    for i in range (num_class):
        if (delta_y >= min_delta+ interval_delta*i) and (delta_y < min_delta+interval_delta*(i+1)):
            return i
        elif (delta_y == min_delta + interval_delta*(i+1)):
            return i
        else:
            print('out of range!')

def custom_objective(p_true, p_pred, num_class, interval_delta):
    lambda_para = 0.1
    minus_delta = - interval_delta
    plus_delta = interval_delta
    l_a = 2/(minus_delta*(minus_delta-plus_delta)*interval_delta)
    print('l_a:', l_a)
    l_b = 2/(minus_delta*plus_delta*interval_delta)
    print('l_b:', l_b)
    l_c = 2/(plus_delta*(plus_delta-minus_delta)*interval_delta)
    print('l_c:', l_c)
    input = np.ones(num_class-2)*interval_delta
    print('input:', input)
    D = np.diag(input)
    # print('D:', D, D.shape)

    L = np.zeros((num_class-2,num_class))
    # print('L:', L, L.shape)
    for i in range(num_class-2):
        L[i, i] = l_a
        L[i, i+1] = l_b
        L[i, i+2] = l_c
    print('L:', L, L.shape)

    # '''Just another crossentropy'''
    # y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # y_pred /= y_pred.sum(axis=-1, keepdims=True)
    internal_matrix = np.dot(L, p_pred)
    cce = T.nnet.categorical_crossentropy(p_pred, p_true) + lambda_para*np.dot((np.dot(np.transpose(internal_matrix),D), internal_matrix)
    print('cce: ', cce, type(cce))
    return cce

custom_objective(30, 1.0199)