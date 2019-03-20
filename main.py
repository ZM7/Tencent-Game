from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
import os
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not p in sys.path:
    sys.path.append(p)
from utils import Data
from data_process_fnn_concat import fnn_seq, read_embedding
import time
from models import FNN
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sample', action='store_true', help='dataset')
parser.add_argument('--baseline', action='store_true', help='base_line')
parser.add_argument('--method', type=str, default='line', help='line/deepWalk/node2vec')
parser.add_argument('--min_round', type=int, default=1)
parser.add_argument('--num_round', type=int, default=500)
parser.add_argument('--early_stop_round', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=512)
opt = parser.parse_args()
print(opt)
field_sizes = [6056, 88561, 100, 100, 100, 100, 100]
if opt.baseline:
    field_sizes = [6056, 88561]
OUTPUT_DIM = 1
field_offsets = [sum(field_sizes[:i]) for i in range(len(field_sizes))]
input_dim = sum(field_sizes)


min_round = opt.min_round
num_round = opt.num_round
early_stop_round = opt.early_stop_round
batch_size = opt.batch_size

if opt.sample:
    path = 'sample'
else:
    path = 'all_data'

app_file = './Dataset/item_pro.pickle'
user_file = './Dataset/user_pro.pickle'
train_file = './Dataset/' + path + '/fnn_train.pickle'
test_file = './Dataset/' + path + '/fnn_test.pickle'
seq_file = './Dataset/all_data/aliases_dict.pickle'
embed_file = './Dataset/all_data/'+opt.method+'_embedding.npy'
path_log = './Dataset/login.txt'
path_pay = './Dataset/pay.txt'

user = pickle.load(open(user_file, 'rb'))
app = pickle.load(open(app_file, 'rb'))
dic = pickle.load(open(seq_file, 'rb'))
#embed = np.load(embed_file)
embed = read_embedding(opt)
# train_seq = pickle.load(open(train_file,'rb'))
# test_seq = pickle.load(open(test_file,'rb'))
train_seq, test_seq = fnn_seq(path_log ,path_pay,opt.sample,win=4)
print('train data size:', len(train_seq))
print('test data size:', len(test_seq))
train_data = Data(train_seq, user, app, embed, dic, opt.batch_size, field_offsets, opt.baseline, shuffle=True)
test_data = Data(test_seq, user, app, embed, dic, opt.batch_size, field_offsets, opt.baseline, shuffle=False)
del train_seq
del test_seq
del user
del app
del dic
del embed

algo = 'fnn'
if algo == 'fnn':
    fnn_params = {
        'field_sizes': field_sizes,
        'embed_size': 64,
        'layer_sizes': [64,64,64,64,64,64,64,1],
        'layer_acts': ['relu', 'relu','relu','relu','relu', 'relu','relu',None],
        'drop_out': [0, 0, 0, 0,0, 0, 0,0],
        'opt_algo': 'gd',
        'learning_rate': 0.0001,
        'embed_l2': 0,
        'layer_l2': [0, 0, 0, 0,0, 0,0,0],
        'random_seed': 0
    }
    print(fnn_params)
    model = FNN(**fnn_params)

def train(model):
    history_score = []
    for i in range(num_round):
        fetches = [model.optimizer, model.loss, model.y_prob]
        ls = []
        # bar = progressbar.ProgressBar()
        print('[%d]\ttraining...' % i)
        print(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        train_preds = []
        train_true = []
        for train_array in train_data.generate_batch():
            X_i, y_i = train_array
            _, l, preds = model.run(fetches, X_i, y_i)
            ls.append(l)
            train_preds.extend(preds)
            train_true.extend(y_i)
        print('[%d]\tevaluating...' % i)
        print(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
        test_preds = []
        test_true = []
        for test_array in test_data.generate_batch():
            X_i, y_i = test_array
            preds = model.run(model.y_prob, X_i, mode='test')
            test_preds.extend(preds)
            test_true.extend(y_i)
        train_score = roc_auc_score(train_true, train_preds)
        test_score = roc_auc_score(test_true, test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                        -1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break

if __name__ == '__main__':
    train(model)
