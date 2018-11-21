import sys
import os
args = {}
args['max_length'] = 'ALL'
args['task']='PRED'
args['n_hidden'] = '100'
args['samples']='ALL'
args['datasets'] = ['data/github/github','data/dota/dota_class','data/dota/dota','data/freecodecamp_students/freecodecamp_students','data/reddit/reddit','data/reddit_comments/reddit_comments']
for st in sys.argv[1:]:
    splt = st.index('=')
    key = st[:splt]
    if key == 'datasets':
        val = [x for x in st[splt+1:].split(',')]
    else:
        val = st[splt+1:]
    args[key]=val
for ds in args['datasets']:
    print(ds)
    argstr = ''
    if 'class' in ds:
        args['task'] = 'CLASS'
    else:
        args['task'] = 'PRED'
    for key in args.keys():
        if key != 'datasets':
            argstr+=' '+str(key)+'='+str(args[key])

    if ds!='data/github/github':
        argstr += ' dataset='
        argstr+=ds
        cwd = os.path.join(os.getcwd(), "run/run.py" +argstr)
        os.system('{} {}'.format('python3', cwd))
    else:
        cwd = os.path.join(os.getcwd(), "run/run_github.py" +argstr)
        os.system('{} {}'.format('python3', cwd))
