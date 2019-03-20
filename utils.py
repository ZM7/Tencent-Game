from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from scipy.sparse import coo_matrix, csr_matrix, vstack, hstack
import sys
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl
import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix
DTYPE = tf.float32

STDDEV = 1e-3
MINVAL = -1e-3
MAXVAL = 1e-3


def shuffle(data):
    X, y = data
    ind = np.arange(X.shape[0])
    for i in range(7):
        np.random.shuffle(ind)
    return X[ind], y[ind]


def csr_2_input(csr_mat):
    if not isinstance(csr_mat, list):
        coo_mat = csr_mat.tocoo()
        indices = np.vstack((coo_mat.row, coo_mat.col)).transpose()
        values = csr_mat.data
        shape = csr_mat.shape
        return indices, values, shape
    else:
        inputs = []
        for csr_i in csr_mat:
            inputs.append(csr_2_input(csr_i))
        return inputs


def slice(csr_data):
    if not isinstance(csr_data[0], list):
        slc_data = csr_data[0]
        slc_labels = csr_data[1]
    else:
        slc_data = []
        for d_i in csr_data[0]:
            slc_data.append(d_i)
        slc_labels = csr_data[1]
    return csr_2_input(slc_data), slc_labels


def split_data(data, FIELD_OFFSETS, skip_empty=True):
    fields = []
    for i in range(len(FIELD_OFFSETS) - 1):
        start_ind = FIELD_OFFSETS[i]
        end_ind = FIELD_OFFSETS[i + 1]
        if skip_empty and start_ind == end_ind:
            continue
        field_i = data[0][:, start_ind:end_ind]
        fields.append(field_i)
    fields.append(data[0][:, FIELD_OFFSETS[-1]:])
    return fields, data[1]


def read_data(user, app, embed, seq, dic, start=0, size=-1, baseline=False):
    X = []
    y = []
    if size == -1 or start + size >= len(seq):
        seq = seq[start:]
    else:
        seq = seq[start:start + size]
    if baseline:
        for i in range(len(seq)):
            ap_emb = app[seq[i][2]]
            us_emb = user[seq[i][1]]
            X.append(hstack([us_emb, ap_emb]))
            y.append(seq[i][3])

    else:
        for i in range(len(seq)):
            ap_emb = app[seq[i][2]]
            us_emb = user[seq[i][1]]
            vec = [ap_emb, us_emb, csr_matrix(embed[dic[seq[i][2]]])]
            for m_seq in seq[i][0]:
                if m_seq == 0:
                    vec.append(csr_matrix(embed[0]))
                else:
                    seq_= [dic[us] for us in m_seq]
                    vec.append(csr_matrix(embed[seq_].mean(0)))
            X.append(hstack(vec))
            y.append(seq[i][3])

    X = vstack(X).tocsr()
    y = np.reshape(np.array(y), [-1])
    return X, y


class Data(object):
    def __init__(self, data, user, item, embed, aliases_dict, batch_size, FIELD_OFFSETS, baseline=False, shuffle=True):
        self.user = user
        self.item = item
        self.embed = embed
        self.data = np.asarray(data)
        self.batch_size = batch_size
        self.length = len(data)
        self.FIELD_OFFSETS = FIELD_OFFSETS
        self.aliases_dic = aliases_dict
        self.baseline = baseline
        self.shuffle = shuffle

    def generate_batch(self):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.data = self.data[shuffled_arg]
        for i in range(int(self.length/self.batch_size+1)):
            batch = read_data(self.user, self.item, self.embed, self.data, self.aliases_dic,
                              i*self.batch_size, self.batch_size, self.baseline)
            batch = split_data(batch, self.FIELD_OFFSETS)
            X, y = slice(batch)
            yield [X, y]


def init_var_map(init_vars, init_path=None):
    if init_path is not None:
        load_var_map = pkl.load(open(init_path, 'rb'))
        print('load variable map from', init_path, load_var_map.keys())
    var_map = {}
    for var_name, var_shape, init_method, dtype in init_vars:
        if init_method == 'zero':
            var_map[var_name] = tf.Variable(tf.zeros(var_shape, dtype=dtype), name=var_name, dtype=dtype)
        elif init_method == 'one':
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype), name=var_name, dtype=dtype)
        elif init_method == 'normal':
            var_map[var_name] = tf.Variable(tf.random_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'tnormal':
            var_map[var_name] = tf.Variable(tf.truncated_normal(var_shape, mean=0.0, stddev=STDDEV, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'uniform':
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=MINVAL, maxval=MAXVAL, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif init_method == 'xavier':
            maxval = np.sqrt(6. / np.sum(var_shape))
            minval = -maxval
            var_map[var_name] = tf.Variable(tf.random_uniform(var_shape, minval=minval, maxval=maxval, dtype=dtype),
                                            name=var_name, dtype=dtype)
        elif isinstance(init_method, int) or isinstance(init_method, float):
            var_map[var_name] = tf.Variable(tf.ones(var_shape, dtype=dtype) * init_method, name=var_name, dtype=dtype)
        elif init_method in load_var_map:
            if load_var_map[init_method].shape == tuple(var_shape):
                var_map[var_name] = tf.Variable(load_var_map[init_method], name=var_name, dtype=dtype)
            else:
                print('BadParam: init method', init_method, 'shape', var_shape, load_var_map[init_method].shape)
        else:
            print('BadParam: init method', init_method)
    return var_map


def activate(weights, activation_function):
    if activation_function == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif activation_function == 'softmax':
        return tf.nn.softmax(weights)
    elif activation_function == 'relu':
        return tf.nn.relu(weights)
    elif activation_function == 'tanh':
        return tf.nn.tanh(weights)
    elif activation_function == 'elu':
        return tf.nn.elu(weights)
    elif activation_function == 'none':
        return weights
    else:
        return weights


def get_optimizer(opt_algo, learning_rate, loss):
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)



