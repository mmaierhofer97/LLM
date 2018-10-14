import os
import sys
import random
filepath = "'data/dota/dota"
args = ''
if len(sys.argv)>1:
	for a in sys.argv[1:]:
		args += ' '+a
print(args)
for i in range(10):
	print(i)
	cwd = os.path.join(os.getcwd(), "main.py dataset="+filepath+"' 'encoder=LLM' 'device=GPU' 'task=PRED' 'model_load_name=FALSE'"+args)
	os.system('{} {}'.format('python3', cwd))
	cwd = os.path.join(os.getcwd(), "main.py dataset="+filepath+"' 'encoder=LSTM' 'device=GPU' 'task=PRED' 'model_load_name=FALSE'"+args)
	os.system('{} {}'.format('python3', cwd))
