import os
import sys
filepath = "'data/synth_accum/accum_pred"
filenames = ["1000'","300'","100'","30'","10'"]
for i in range(10):
	print(i)
	for filename in filenames:
		cwd = os.path.join(os.getcwd(), "main.py "+filepath+filename+" 'LLM' 'GPU' 'PRED'")
		os.system('{} {}'.format('python3', cwd))
		cwd = os.path.join(os.getcwd(), "main.py "+filepath+filename+" 'LSTM' 'GPU' 'PRED'")
		os.system('{} {}'.format('python3', cwd))