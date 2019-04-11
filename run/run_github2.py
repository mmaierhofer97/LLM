
import os
from inspect import getsourcefile
import os.path
import sys

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

import sys
import random
import data_help as DH
args = {}
args['max_length'] = 'ALL'
args['encoder']='LLM'
args['device']='GPU'
args['task']='PRED'
args['model_load_name']='FALSE'
args['gdataset'] = 'data/github/github'
args['n_hidden'] = '100'
for st in sys.argv[1:]:
    splt = st.index('=')
    key = st[:splt]
    val = st[splt+1:]
    args[key]=val
for i in range(10):
    print(i)
    r = str(random.randint(0,99)).zfill(2)
    args['seed'] = random.randint(1,10**8)
    args['encoder'] = 'LLM2'
    args['dataset'] = args['gdataset']+r
    argstr = ''
    for key in args.keys():
        argstr+=' '+str(key)+'='+str(args[key])
    cwd = os.path.join(os.getcwd(), "main.py" +argstr)
    os.system('{} {}'.format('python3', cwd))
    args['encoder'] = 'LLM'
    argstr = ''
    for key in args.keys():
        argstr+=' '+str(key)+'='+str(args[key])
    cwd = os.path.join(os.getcwd(), "main.py"+argstr)
    os.system('{} {}'.format('python3', cwd))
    LLMfile = open(args['dataset']+'tmp_LLM2'+args['n_hidden']+'.txt','rt')
    LSTMfile = open(args['dataset']+'tmp_LLM'+args['n_hidden']+'.txt','rt')
    a = [float(i) for i in LLMfile.read()[:-2].split(',')]
    b =  [float(i) for i in LSTMfile.read()[:-2].split(',')]
    ml = str(args['max_length'])
    DH.write_history([a[1],b[1]],args['dataset']+'_'+str(ml)+'_paired_both_test'+args['n_hidden']+'.txt', i, False)
    DH.write_history([a[0],b[0]],args['dataset']+'_'+str(ml)+'_paired_both_train'+args['n_hidden']+'.txt', i, False)
    DH.write_history([a[2],b[2]],args['dataset']+'_'+str(ml)+'_paired__both_valid'+args['n_hidden']+'.txt', i, False)
    LLMfile = open(args['dataset']+'tmp_auc_LLM2'+args['n_hidden']+'.txt','rt')
    LSTMfile = open(args['dataset']+'tmp_auc_LLM'+args['n_hidden']+'.txt','rt')
    a = [float(i) for i in LLMfile.read()[:-2].split(',')]
    b =  [float(i) for i in LSTMfile.read()[:-2].split(',')]
    ml = str(args['max_length'])
    DH.write_history([a[1],b[1]],args['dataset']+'_'+str(ml)+'_paired_both_test_auc'+args['n_hidden']+'.txt', i, False)
    DH.write_history([a[0],b[0]],args['dataset']+'_'+str(ml)+'_paired_both_train_auc'+args['n_hidden']+'.txt', i, False)
    DH.write_history([a[2],b[2]],args['dataset']+'_'+str(ml)+'_paired_both_valid_auc'+args['n_hidden']+'.txt', i, False)
