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
args['datasets'] = ['data/freecodecamp_students/freecodecamp_students','data/freecodecamp_students/freecodecamp_students','data/dota/dota','data/dota/dota']
nh = ['100','200','200','50']
args['n_hidden'] = '100'
for st in sys.argv[1:]:
    splt = st.index('=')
    key = st[:splt]
    val = st[splt+1:]
    args[key]=val
for i in range(10):
    j=0
    print(i)
    args['seed'] = random.randint(1,10**8)
    args['encoder'] = 'LLM'
    args['n_hidden'] = nh[j]
    args['dataset'] = args['datasets'][j]
    argstr = ''
    for key in args.keys():
        argstr+=' '+str(key)+'='+str(args[key])
    cwd = os.path.join(os.getcwd(), "main.py" +argstr)
    os.system('{} {}'.format('python3', cwd))
    j=j+1
    args['encoder'] = 'LSTM'
    args['n_hidden'] = nh[j]
    args['dataset'] = args['datasets'][j]
    argstr = ''
    for key in args.keys():
        argstr+=' '+str(key)+'='+str(args[key])
    cwd = os.path.join(os.getcwd(), "main.py" +argstr)
    os.system('{} {}'.format('python3', cwd))
    j=j+1
    LLMfile = open(args['dataset']+'tmp_LLM'+nh[j-2]+'.txt','rt')
    LSTMfile = open(args['dataset']+'tmp_LSTM'+nh[j-1]+'.txt','rt')
    a = [float(i) for i in LLMfile.read()[:-2].split(',')]
    b =  [float(i) for i in LSTMfile.read()[:-2].split(',')]
    ml = str(args['max_length'])
    DH.write_history([a[1],b[1]],args['dataset']+'_'+str(ml)+'_paired_test'+'_mixed'+'.txt', i, False)
    DH.write_history([a[0],b[0]],args['dataset']+'_'+str(ml)+'_paired_train'+'_mixed'+'.txt', i, False)
    DH.write_history([a[2],b[2]],args['dataset']+'_'+str(ml)+'_paired_valid'+'_mixed'+'.txt', i, False)
    LLMfile = open(args['dataset']+'tmp_auc_LLM'+args['n_hidden']+'.txt','rt')
    LSTMfile = open(args['dataset']+'tmp_auc_LSTM'+args['n_hidden']+'.txt','rt')
    a = [float(i) for i in LLMfile.read()[:-2].split(',')]
    b =  [float(i) for i in LSTMfile.read()[:-2].split(',')]
    ml = str(args['max_length'])
    DH.write_history([a[1],b[1]],args['dataset']+'_'+str(ml)+'_paired_test_auc'+'_mixed'+'.txt', i, False)
    DH.write_history([a[0],b[0]],args['dataset']+'_'+str(ml)+'_paired_train_auc'+'_mixed'+'.txt', i, False)
    DH.write_history([a[2],b[2]],args['dataset']+'_'+str(ml)+'_paired_valid_auc'+'_mixed'+'.txt', i, False)

    args['seed'] = random.randint(1,10**8)
    args['encoder'] = 'LLM'
    args['n_hidden'] = nh[j]
    args['dataset'] = args['datasets'][j]
    argstr = ''
    for key in args.keys():
        argstr+=' '+str(key)+'='+str(args[key])
    cwd = os.path.join(os.getcwd(), "main.py" +argstr)
    os.system('{} {}'.format('python3', cwd))
    j=j+1
    args['encoder'] = 'LSTM'
    args['n_hidden'] = nh[j]
    args['dataset'] = args['datasets'][j]
    argstr = ''
    for key in args.keys():
        argstr+=' '+str(key)+'='+str(args[key])
    cwd = os.path.join(os.getcwd(), "main.py" +argstr)
    os.system('{} {}'.format('python3', cwd))
    j=j+1
    LLMfile = open(args['dataset']+'tmp_LLM'+nh[j-2]+'.txt','rt')
    LSTMfile = open(args['dataset']+'tmp_LSTM'+nh[j-1]+'.txt','rt')
    a = [float(i) for i in LLMfile.read()[:-2].split(',')]
    b =  [float(i) for i in LSTMfile.read()[:-2].split(',')]
    ml = str(args['max_length'])
    DH.write_history([a[1],b[1]],args['dataset']+'_'+str(ml)+'_paired_test'+'_mixed'+'.txt', i, False)
    DH.write_history([a[0],b[0]],args['dataset']+'_'+str(ml)+'_paired_train'+'_mixed'+'.txt', i, False)
    DH.write_history([a[2],b[2]],args['dataset']+'_'+str(ml)+'_paired_valid'+'_mixed'+'.txt', i, False)
    LLMfile = open(args['dataset']+'tmp_auc_LLM'+args['n_hidden']+'.txt','rt')
    LSTMfile = open(args['dataset']+'tmp_auc_LSTM'+args['n_hidden']+'.txt','rt')
    a = [float(i) for i in LLMfile.read()[:-2].split(',')]
    b =  [float(i) for i in LSTMfile.read()[:-2].split(',')]
    ml = str(args['max_length'])
    DH.write_history([a[1],b[1]],args['dataset']+'_'+str(ml)+'_paired_test_auc'+'_mixed'+'.txt', i, False)
    DH.write_history([a[0],b[0]],args['dataset']+'_'+str(ml)+'_paired_train_auc'+'_mixed'+'.txt', i, False)
    DH.write_history([a[2],b[2]],args['dataset']+'_'+str(ml)+'_paired_valid_auc'+'_mixed'+'.txt', i, False)
