#import tensorflow as tf
import numpy as np
import data_help as DH
import tensor_classes_helpers as TCH
import sys
from tensorflow.python import debug as tf_debug
import os
from sklearn import metrics
ops = {'datasets' : [], 'samples':'ALL','timescales' : 2.0 ** np.arange(-7,7),'learn_timescales': True}
for st in sys.argv[1:]:
    ops['datasets'].append(st)
for ds in ops['datasets']:
    ends = ['']
    if ds == 'data/github/github':
        ends = [str(x).zfill(2) for x in range(100)]
    print(ds)
    m = 0
    M = -1
    L = 10**10
    for end in ends:
        datasets = DH.load_data(ds+end, sort_by_len=False, samples = ops['samples'])
        train_set = datasets['train_set']
        test_set = datasets['test_set']
        valid_set = datasets['valid_set']
        ops['max_length']=100
        if ops['max_length'] == "ALL" or 'class' in ds:
            ops['max_length'] = DH.longest_seq([train_set,valid_set,test_set]) #Can't concatenate classification data


        m = max(DH.num_classes([train_set,valid_set,test_set],ops['max_length'])-1,m)

        if ops['learn_timescales']==True:
            ops['timescales'] = DH.set_timescales(train_set,(ops['timescales']))
        L = min(ops['timescales'][1],L)
        M = max(ops['timescales'][-2],M)
    print('Max num classes {}'.format(m))
    print('L = {}, M = {}'.format(L,M))
