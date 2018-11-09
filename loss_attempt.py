import theano
import theano.tensor as T
import numpy as np


def create_one_hot_vector(delta_y_value, min_delta, interval_delta, num_class):
    p_true = np.zeros(num_class)
    for i in range (num_class + 1):
        if (delta_y_value >= min_delta+ interval_delta*i) and (delta_y_value < min_delta+interval_delta*(i+1)):
            p_true[i]=1
            print("i:", i)
            print('p_true:', p_true)
            return p_true

epsilon = 1.0e-9
# C-delata function
def C_delta(delta_y, min_delta, interval_delta, num_class):
    print('delta y: ', delta_y)
    if isinstance(delta_y, int or float):
        return create_one_hot_vector(delta_y, min_delta, interval_delta, num_class)
    else:
        return np.array(list([create_one_hot_vector(x, min_delta, interval_delta, num_class) for x in delta_y]))

# y_true = C_delta(0.25, 0.1, 0.05, 30)

def custom_objective(p_true, p_pred, num_class, interval_delta):
# def custom_objective(p_true, p_pred):
#     num_class = 30
#     interval_delta = 1.09
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
    print('internal matrix: ', internal_matrix.shape)
    first_dot = np.dot(np.transpose(internal_matrix), D)
    print('first dot: ', first_dot.shape)
    # print(lambda_para * np.dot(first_dot, internal_matrix))
    cce = T.nnet.categorical_crossentropy(p_pred, p_true) + lambda_para*np.dot(first_dot, internal_matrix)
    print('cce: ', type(cce))
    return cce

# custom_objective(y_true, 30, 1.0199)