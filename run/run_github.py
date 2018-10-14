
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
for st in sys.argv[1:]:
    splt = st.index('=')
    key = st[:splt]
    val = st[splt+1:]
    args[key]=val
for i in range(10):
    print(i)
	r = str(random.randint(0,99)).zfill(2)
    args['seed'] = random.randint(1,10**8)
    args['encoder'] = 'LLM'
	args['dataset'] = args['gdataset']+r
    argstr = ''
    for key in args.keys():
        argstr+=' '+str(key)+'='+str(args[key])
    cwd = os.path.join(os.getcwd(), "main.py" +argstr)
    os.system('{} {}'.format('python3', cwd))
    args['encoder'] = 'LSTM'
    argstr = ''
    for key in args.keys():
        argstr+=' '+str(key)+'='+str(args[key])
    cwd = os.path.join(os.getcwd(), "main.py"+argstr)
    os.system('{} {}'.format('python3', cwd))
    LLMfile = open('records/tmp_LLM.txt','rt')
    LSTMfile = open('records/tmp_LSTM.txt','rt')
    a = float(LLMfile.read())
    b = float(LSTMfile.read())
    ml = str(args['max_length'])
    DH.write_history([a,b],args['gdataset']+'_'+str(ml)+'_paired_test.txt', i, False)
