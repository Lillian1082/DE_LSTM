from math import sqrt
import tensorflow as tf
from keras import Model, Input
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.python.keras import backend as K
import keras.backend as K
# from loss_attempt import custom_objective

num_class= 30

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[ 1 ]
    # print("n_vars:", n_vars)
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [ ('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars) ]
    # print("names1:", names)
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [ ('var%d(t)' % (j + 1)) for j in range(n_vars) ]
        else:
            names += [ ('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars) ]
    # print("names2:", names)
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('Train_3_Test.csv', header=0, index_col=0, low_memory=False)

values = dataset.values
print('values:',values.shape, values[:,:-6] )
max_values = np.max(values[:,:-6])
print('max_values:', max_values)
min_values = np.min(values[:,:-6])
print('min_values:', min_values)
delta1 = max_values - min_values
print('delta1:', delta1)

delta_values = [ ]
for i in range(0, values.shape[ 1 ] - 6):
    for j in range(values.shape[ 0 ] - 1):
        delta = values[ j + 1 ][ i ] - values[ j ][ i ]
        delta_values.append(delta)
delta_values = np.reshape(delta_values, (6, -1)).transpose()
# print('delta_values:', delta_values, np.shape(delta_values))

max_delta = np.max(delta_values)
print('max_delta:', max_delta)
min_delta = np.min(delta_values)
print('min_delta:', min_delta)
range_delta = max_delta - min_delta
print('range_delta:', range_delta)
interval_delta = range_delta/num_class
print('interval_delta:', interval_delta)


# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print("scaled:", scaled, scaled.shape)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# print("reframed:", reframed)
# drop columns we don't want to predict
reframed.drop(reframed.columns[ [ 18, 19, 20, 21, 22, 23 ] ], axis=1, inplace=True)
print('reframed \n', reframed.head(), reframed.shape)

# split into train and test sets
values = reframed.values
n_train_hours = 40 * 24
n_pred_hours = 2 * 24
train = values[ :n_train_hours, : ]
test = values[ n_train_hours: n_train_hours + n_pred_hours, : ]
# print('train:', train.shape)
# print('test:', test.shape)
# split into input and outputs
train_X, train_y = train[ :, :12 ], train[ :, 12: ]
# print('train X: ', train_X.shape)
# print('train Y:', train_y.shape)
test_X, test_y = test[ :, :12 ], test[ :, 12: ]
# print('Test X:', test_X.shape)
# print('Test Y:', test_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[ 0 ], 1, train_X.shape[ 1 ]))
test_X = test_X.reshape((test_X.shape[ 0 ], 1, test_X.shape[ 1 ]))
print('train_X:', train_X.shape, 'train_Y:', train_y.shape, 'test_X:', test_X.shape, 'test_Y:', test_y.shape)

# design network
# x = Dense(train_X.shape[2], input_shape=(train_X.shape[1], train_X.shape[2]), activation= 'tanh', use_bias= True)(train_X)
# x = Dense(train_X.shape[2], activation= 'tanh', use_bias= True)(x)
# h = tf.zeros([10, 50])
main_input = Input(shape=(train_X.shape[ 1 ], train_X.shape[ 2 ]), batch_shape=(24, 1, 12), name='main_input')
# auxiliary_input = Input(shape=(1, 50), name='auxiliary_input')
x = Dense(24, activation='tanh')(main_input)
x = Dense(24, activation='tanh')(x)
# print('x:', x, x.shape)
# x = x + Dense(10)(auxiliary_input)
# print('x:', x, x.shape)
h = LSTM(50, stateful=True, name='LSTM')(x)
# print('h:', h, h.shape)

prediction = Dense(24, activation='tanh')(h)
# print('h:', h, h.shape)
prediction = Dense(6)(prediction)
print('prediction:', prediction, prediction.shape)

# auxiliary_array = tf.reshape(h, [1,10])

# print('auxiliary_array:', auxiliary_array)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# K.set_session(sess)
# new_array = auxiliary_array.eval(session = sess)
# print('new_array:', new_array)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# auxiliary_array = h.eval(session = sess)
# print('auxiliary_array:', auxiliary_array, auxiliary_array.shape)
# auxiliary_array = auxiliary_array.reshape(h.shape[0], 1, 50)

# print('main input: ', main_input)
# print('auxiliary input: ', auxiliary_input)
# print('predictions: ', prediction)
# model = Model(inputs=[main_input, auxiliary_input], outputs=prediction)
model = Model(inputs=main_input, outputs=prediction)
model.reset_states()
# intermediate_output = np.zeros([10, 50])
# layer_name = 'intermediate_layer'
# intermediate_layer_model = Model(input=model.input,
#                                  output=model.get_layer('LSTM').output)
# intermediate_output = intermediate_layer_model.predict([train_X, intermediate_output])
#
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[ 'accuracy' ])
# fit network
# history = model.fit( [train_X, intermediate_output], train_y, epochs=50, batch_size=72 , verbose=2,
#                      shuffle=False)
history = model.fit(train_X, train_y, epochs=400, batch_size=24, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# print('h:', h, h.shape)

# plot history
pyplot.plot(history.history[ 'loss' ], label='train')
# print('what is history.history', history.history)
pyplot.plot(history.history[ 'val_loss' ], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X, batch_size=24)
# print('the just predicted yhat: ', yhat.shape)
test_X = test_X.reshape((test_X.shape[ 0 ], test_X.shape[ 2 ]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[ :, 6: ]), axis=1)
# print('inv yhat: ', inv_yhat.shape)
inv_yhat = scaler.inverse_transform(inv_yhat)
# print('inv yhat After: ', inv_yhat, inv_yhat.shape)
inv_yhat = inv_yhat[ :, :6 ]
# print('inv yhat 3: ', inv_yhat, inv_yhat.shape)
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 6))

inv_y = concatenate((test_y, test_X[ :, 6: ]), axis=1)
# print('inv_y:', inv_y, inv_y.shape)
inv_y = scaler.inverse_transform(inv_y)
# print('inv_y After:', inv_y, inv_y.shape)
inv_y = inv_y[ :, :6 ]
# print('inv_y 3:', inv_y, inv_y.shape)
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

for i in range(6):
    pyplot.plot(inv_y[ :, i ], label='ground truth%d' % i)
    pyplot.plot(inv_yhat[ :, i ], label='prediction%d' % i)
    pyplot.legend()
    pyplot.show()
