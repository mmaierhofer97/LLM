import sys
import os
args = {}
args['max_length'] = 'ALL'
args['task']='PRED'
args['n_hidden'] = '100'
args['samples']='ALL'
args['datasets'] = ['data/github/github','data/dota/dota_class','data/dota/dota','data/freecodecamp_students/freecodecamp_students','data/reddit/reddit','data/reddit_comments/reddit_comments','data/quizlet/quizlet']
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
    if 'class' in ds or 'rhythm' in ds:
        args['task'] = 'CLASS'
    elif 'quizlet' in ds:
        args['task'] = 'PRED_CORR'
    else:
        args['task'] = 'PRED'
    for key in args.keys():
        if key != 'datasets':
            argstr+=' '+str(key)+'='+str(args[key])
    special_case = ''
    cases = ['github','quizlet']
    for case in cases:
        if case in ds:
            special_case = case

    if special_case!='':
        cwd = os.path.join(os.getcwd(), "run/run_"+special_case+"2.py" +argstr)
        os.system('{} {}'.format('python3', cwd))

    else:
        argstr += ' dataset='
        argstr+=ds
        cwd = os.path.join(os.getcwd(), "run/run2.py" +argstr)
        os.system('{} {}'.format('python3', cwd))
