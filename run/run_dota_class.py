import os
import sys
import random
filepath = "'data/dota/dota_class"
args = ''
if len(sys.argv)>1:
	for a in sys.argv[1:]:
		args += ' '+a 
print(args)
for i in range(10):
	print(i)
	cwd = os.path.join(os.getcwd(), "main.py "+filepath+"' 'LLM' 'GPU' 'CLASS' 'FALSE'"+args)
	print(cwd)
	os.system('{} {}'.format('python3', cwd))
	cwd = os.path.join(os.getcwd(), "main.py "+filepath+"' 'LSTM' 'GPU' 'CLASS' 'FALSE'"+args)
	os.system('{} {}'.format('python3', cwd))
