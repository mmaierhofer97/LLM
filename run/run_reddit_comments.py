import os
import sys
import random
filepath = "'data/reddit_comments/reddit_comments"
args = ''
if len(sys.argv)>1:
	for a in sys.argv[1:]:
		args += ' '+a 
print(args)
for i in range(10):
	print(i)
	cwd = os.path.join(os.getcwd(), "main.py "+filepath+"' 'LLM' 'GPU' 'PRED' 'FALSE'"+args)
	print(cwd)
	os.system('{} {}'.format('python3', cwd))
	cwd = os.path.join(os.getcwd(), "main.py "+filepath+"' 'LSTM' 'GPU' 'PRED' 'FALSE'"+args)
	os.system('{} {}'.format('python3', cwd))
