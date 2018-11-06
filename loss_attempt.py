import theano
import theano.tensor as T

epsilon = 1.0e-9
def custom_objective(y_true, y_pred):

    '''Just another crossentropy'''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= y_pred.sum(axis=-1, keepdims=True)
    cce = T.nnet.categorical_crossentropy(y_pred, y_true)
    print('type of cce: ', type(cce))
    return cce