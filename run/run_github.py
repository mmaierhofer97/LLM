import os
import sys
import random
filepath = "'data/github/github"
args = ''
if len(sys.argv)>1:
	for a in sys.argv[1:]:
		args += ' '+a 
print(args)
for i in range(10):
	print(i)
	r = str(random.randint(0,99)).zfill(2)
	cwd = os.path.join(os.getcwd(), "main.py "+filepath+r+"' 'LLM' 'GPU' 'PRED' 'FALSE'"+args)
	print(cwd)
	os.system('{} {}'.format('python3', cwd))
	cwd = os.path.join(os.getcwd(), "main.py "+filepath+r+"' 'LSTM' 'GPU' 'PRED' 'FALSE'"+args)
	os.system('{} {}'.format('python3', cwd))
