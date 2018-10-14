import os
import sys
filepath = "'data/synth_accum/accum_pred"
filenames = ["1000'","300'","100'","30'","10'"]
for i in range(10):
	print(i)
	for filename in filenames:
	cwd = os.path.join(os.getcwd(), "main.py dataset="+filepath+"' 'encoder=LLM' 'device=GPU' 'task=PRED' 'model_load_name=FALSE'"+args)
	os.system('{} {}'.format('python3', cwd))
	cwd = os.path.join(os.getcwd(), "main.py dataset="+filepath+"' 'encoder=LSTM' 'device=GPU' 'task=PRED' 'model_load_name=FALSE'"+args)
	os.system('{} {}'.format('python3', cwd))
