# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# """
# Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
#
# Run this script on tensorflow r0.10. Errors appear when using lower versions.
# """
from typing import List, Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pandas import DataFrame
from tensorflow.python.framework.test_ops import none

BATCH_START = 0
# TIME_STEPS = 20
BATCH_SIZE = 50 # The meaning is time_steps
INPUT_SIZE = 10
OUTPUT_SIZE = 10 # K the number of classes
CELL_SIZE = 10
LR = 0.006
MAX_VALUE = 0
MIN_VALUE = 0
NUM_EXTRA_FACTORS = 4

data = pd.read_csv("Train_2.csv").set_index('dateTime')
# data = pd.read_csv(url, sep=";", index_col=0, parse_dates=True, decimal=',').set_index('dateTime')
# print ('data:')
# print (data)
data.index = pd.to_datetime(data.index)
# print ('data.index:', data.index, data.index.shape)
# print("data.shape")
# print(data.shape)

num_timeseries = data.shape[1]  # gives number of col count
len_timeseries = data.shape[0]
# print ('num_timeseries:', num_timeseries)
# print ('len_timeseries:', len_timeseries)


timeseries = []

for i in range(0, num_timeseries - 2):  # 除去最后两个出现异常数据的列，除去4个外生变量
    # print("data iloc: ", type(data.iloc[:, i]))
    timeseries.append(data.iloc[:, i].tolist())

for i in range(NUM_EXTRA_FACTORS, num_timeseries - 2):
    if MAX_VALUE < max(timeseries[i][:]):
        MAX_VALUE = max(timeseries[i][:])
    if MIN_VALUE > min(timeseries[i][:]):
        MIN_VALUE = min(timeseries[i][:])

# print("MAX_VALUE:", MAX_VALUE)
# print("MIN_VALUE:", MIN_VALUE)

# print("timeseries:",timeseries)
# print("len(timeseries):", len(timeseries))
# print(len(timeseries[0]))

def plot_timeseries(timeseries):
    fig, axs = plt.subplots(1, 1, figsize=(20, 20), sharex=True)
    axx = axs.ravel()
    # print (axx)
    for i in range(0, num_timeseries - 2):
        timeseries[ i ].loc[ "2017/8/3  10:00:00 AM":"2017/10/4  2:00:00 PM" ].plot(ax=axx[ i ])
        axx[ i ].set_xlabel("dateTime")
        axx[ i ].set_ylabel(timeseries[ i ].name)
        axx[ i ].grid(which='minor', axis='x')
    # df = pd.DataFrame(timeseries, index = pd.date_range("2017/8/3  10:00:00 AM","2017/10/4  2:00:00 PM"))  # type: DataFrame
    # df = df.cumsum()
    # plt.figure(); df.plot(); plt.legend(loc = 'best')
    plt.show()

def convert_to_tensor(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

class LSTMRNN(object):
    def __init__(self, input_size, output_size, cell_size, batch_size, max_value, min_value, num_extra_factors):
        # self.input_size = input_size
        self.input_size = input_size
        self.output_size = output_size # K the number of classes
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.batch_start = 0
        self.num_extra_factors = num_extra_factors

        # seq = [] #the observed value of yt
        # ex_factor = [] #ut
        self.res = []  # the predicted value of yt
        self.pre_input = []  # xt that is combined by yt and ut
        self.ready_input = []  # zt computed by xt and add_input_layer
        self.internal_output = np.zeros((10,1))  # ht computed by add_cell
        # print("internal_output_type:", type(self.internal_output))
        self.final_output = []  # P(t+1) computed by add_output_layer

        self.max_value = max_value
        self.min_value = min_value

        # with tf.name_scope('inputs'):
        #     self.xs = tf.placeholder(tf.float32, [None, time_steps, input_size], name='xs')
        #     self.ys = tf.placeholder(tf.float32, [None, time_steps, output_size], name='ys')
        # with tf.variable_scope('in_hidden'):
        #     self.add_input_layer()
        # with tf.variable_scope('LSTM_cell'):
        #     self.add_cell()
        # with tf.variable_scope('out_hidden'):
        #     self.add_output_layer()
        # with tf.name_scope('cost'):
        #     self.compute_cost()
        # with tf.name_scope('train'):
        #     self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)  # one kind of gradient descent method

        self.alpha_list = []
        self.Ws_in_1 = np.random.random([self.input_size, 1])
        self.bs_in_1 = np.random.random([self.input_size, 1])
        self.Ws_in_2 = np.random.random([self.input_size, 1])
        self.bs_in_2 = np.random.random([self.input_size, 1])
        self.Ws_in_3 = np.random.random([self.input_size, 1])
        self.bs_in_3 = np.random.random([self.input_size, 1])
        self. Ws_out_1 = np.random.random([self.input_size, 1])
        self.bs_out_1 = np.random.random([self.input_size, 1])
        self.Ws_out_2 = np.random.random([self.input_size, 1])
        self.bs_out_2 = np.random.random([self.input_size, 1])

        print("Ws_in_1:", self.Ws_in_1)

    def get_batch(self):
        # global BATCH_START, TIME_STEPS
        # xs shape (50batch, 20steps)
        xs = np.arange(self.batch_start, self.batch_start + self.batch_size)
        # print("xs:", xs)
        for i in range(NUM_EXTRA_FACTORS, num_timeseries - 2):
            self.pre_input.append(timeseries[ i ][self.batch_start: self.batch_start + self.batch_size])
            self.res.append(timeseries[ i ][self.batch_start + 1: self.batch_start + self.batch_size + 1])
        for i in range(0, NUM_EXTRA_FACTORS):
            self.pre_input.append(timeseries[ i ][self.batch_start: self.batch_start + self.batch_size])
        print("pre_input:", self.pre_input)
        print(np.shape(self.pre_input))

        # print("res:", self.res)
        # self.res = convert_to_tensor(self.res)
        # print("res:", self.res)
        #
        # self.input_size = len(self.pre_input)
        # print("input_size:")
        # print(self.input_size)
        # #
        # plt.plot(xs[0][:], pre_input[0][BATCH_START: BATCH_START + TIME_STEPS], 'r')
        # plt.plot(xs[0][:], pre_input[1][BATCH_START: BATCH_START + TIME_STEPS], 'y')
        # plt.plot(xs[0][:], pre_input[2][BATCH_START: BATCH_START + TIME_STEPS], 'g')
        # plt.plot(xs[0][:], pre_input[3][BATCH_START: BATCH_START + TIME_STEPS], 'b')
        # plt.plot(xs[0][:], pre_input[4][BATCH_START: BATCH_START + TIME_STEPS], 'c')
        # plt.plot(xs[0][:], pre_input[5][BATCH_START: BATCH_START + TIME_STEPS], 'm')
        # plt.plot(xs[0][:], pre_input[6][BATCH_START: BATCH_START + TIME_STEPS], 'k--') # Temperature
        # # plt.plot(xs[0][:], pre_input[7][BATCH_START: BATCH_START + TIME_STEPS], 'b:') # Humidity
        # plt.plot(xs[0][:], pre_input[8][BATCH_START: BATCH_START + TIME_STEPS], 'g*') # Dew.Point
        # plt.plot(xs[0][:], pre_input[9][BATCH_START: BATCH_START + TIME_STEPS], 'r+') # Wind.Speed
        # # plt.plot(xs[0][:], res[0][BATCH_START: BATCH_START + TIME_STEPS], 'b--')
        # plt.legend()
        # plt.show()

        self.batch_start += self.batch_size
        # returned pre_input, res and xs: shape (batch, step, input)
        # return [pre_input, res, xs]

    def add_input_layer(self):
        l_in_y = np.multiply(self.Ws_in_1, self.pre_input) + self.bs_in_1
        l_in_y = np.tanh(l_in_y)
        l_in_y = np.multiply(self.Ws_in_2, l_in_y) + self.bs_in_2
        l_in_h = np.multiply(self.Ws_in_3, self.internal_output) + self.bs_in_3
        self.ready_input = l_in_y + l_in_h
        self.ready_input=self.ready_input[np.newaxis,:,:]
        self.ready_input = convert_to_tensor(self.ready_input)
        print("ready_input:", self.ready_input)


    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        # lstm_cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell')
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            # print("what is l_in_y: ", self.l_in_y)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.ready_input, initial_state=self.cell_init_state, time_major=False)
        # print("cell_outputs:", self.cell_outputs)
        # print("cell_final_state:",self.cell_final_state)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, 1], name='2_2D')
        # print('l_out_x:', l_out_x)
        with tf.name_scope('Wx_out_plus_b_1'):
            # Ws (in_size, cell_size)
            Ws_out_1 = self._weight_variable([1, self.output_size],"weights_out_1")
            # print("WS_out_1", Ws_out_1)
            # bs (cell_size, )
            bs_out_1 = self._bias_variable([1, self.output_size],"biases_out_1")
            # print("bS_out_1", bs_out_1)
            # l_out_y = (batch * n_steps, cell_size)
            l_out_y = tf.matmul(l_out_x, Ws_out_1) + bs_out_1
            # print("l_out_y", l_out_y)
        with tf.name_scope('tanh'):
            l_out_y = tf.tanh(l_out_y)
        with tf.name_scope('Wx_out_plus_b_2'):
            # Ws (in_size, cell_size)
            Ws_out_2 = self._weight_variable([self.output_size, self.output_size],"weights_out_2")
            # print("WS_out_2", Ws_out_2)
            # bs (cell_size, )
            bs_out_2 = self._bias_variable([1, self.output_size],"biases_out_2")
            # print("bS_out_2", bs_out_2)
            # l_out_y = (batch * n_steps, cell_size)
            l_out_y = tf.matmul(l_out_y,Ws_out_2) + bs_out_2
        with tf.name_scope('sigmoid'):
            l_out_y = tf.sigmoid(l_out_y)
        # print("l_out_y:", l_out_y)

        # shape = (batch, steps, cell_size, output_size)
        # self.final_output = tf.reshape(l_out_y, [-1, self.time_steps, self.cell_size, self.output_size], name = '2_4D')
        # print("final_output:", self.final_output)

    def compute_cost(self):
        # delta_y = self.max_value - self.min_value
        # big_delta = delta_y / self.output_size
        # delta_i_neg = -big_delta
        # delta_i_pos = big_delta
        #
        # l_ia = (2 / delta_i_neg * (delta_i_neg - delta_i_pos)) * (1 / big_delta)
        # l_ib = (2 / delta_i_neg * delta_i_pos) * (1 / big_delta)
        # l_ic = (2 / delta_i_pos * (delta_i_pos - delta_i_neg)) * (1 / big_delta)
        #
        # L_matrix = []
        # for i in range(self.output_size - 2):
        #     a_new_row = np.ones(shape=[self.output_size,])
        #     a_new_row[i] = l_ia
        #     a_new_row[i+1] = l_ib
        #     a_new_row[i+2] = l_ic
        #     L_matrix.append(a_new_row)
        # L_matrix = np.array(L_matrix)
        # print("L matrix: ", L_matrix)
        # D_matrix = np.identity(self.output_size) * big_delta
        # print("D matrix: ", D_matrix)

        # for i in range (self.output_size) :


        # tf.reshape(self.final_output, [-1], name='reshape_pred')
        # print("final_output:", tf.reshape(self.final_output, [-1]))


        # losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #     [tf.reshape(self.pre_input[0, len(timeseries) - 4], [-1], name='reshape_pred')],
        #     [tf.reshape(self.res, [-1], name='reshape_target')],
        #     [tf.ones([self.batch_size * self.time_steps], dtype=tf.float32)],
        #     average_across_timesteps=True,
        #     softmax_loss_function=self.ms_error,
        #     name='losses'
        # ) + np.diff(self.final_output, 2)

        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    # with tf.variable_scope('weights') as scope:
    #     scope.reuse_variables()
    #     def _weight_variable(self, shape, name = "weights"):
    #         initializer = tf.random_normal_initializer(mean=0., stddev=1., )
    #         return tf.get_variable(name = name, shape = shape, initializer=initializer)
    #
    # with tf.variable_scope('biases') as scope:
    #     scope.reuse_variables()
    #     def _bias_variable(self, shape, name='biases'):
    #         initializer = tf.constant_initializer(0.1)
    #         return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    model = LSTMRNN(INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, MAX_VALUE, MIN_VALUE, NUM_EXTRA_FACTORS)
    sess = tf.Session()
    # with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        print("are we here?")
        init = tf.initialize_all_variables()
    else:
        print("or here")
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    # plot_timeseries(timeseries)

    # plt.ion()
    # plt.show()
    model.get_batch()
    model.add_input_layer()
    model.add_cell()
    model.add_output_layer()
    model.compute_cost()

    # for i in range(200):
    #     seq, res, xs = get_batch()
    #     if i == 0:
    #         feed_dict = {
    #                 model.xs: seq,
    #                 model.ys: res,
    #                 # create initial state
    #         }
    #     else:
    #         feed_dict = {
    #             model.xs: seq,
    #             model.ys: res,
    #             model.cell_init_state: state    # use last state as the initial state for this run
    #         }
    #
    #     _, cost, state, pred = sess.run(
    #         [model.train_op, model.cost, model.cell_final_state, model.pred],
    #         feed_dict=feed_dict)
    #
    #     # plotting
    #     plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    #     plt.ylim((-1.2, 1.2))
    #     plt.draw()
    #     plt.pause(0.3)
    #
    #     if i % 20 == 0:
    #         print('cost: ', round(cost, 4))
    #         result = sess.run(merged, feed_dict)
    #         writer.add_summary(result, i)
