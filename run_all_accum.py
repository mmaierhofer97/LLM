import os
import sys
filepath = "'data/synth_accum/accum_pred"
filenames = ["16'","4'","1'","0.25'","0.0625'"]
for i in range(10):
	print(i)
	for filename in filenames:
		cwd = os.path.join(os.getcwd(), "main.py "+filepath+filename+" 'LLM' 'GPU'")
		os.system('{} {}'.format('python3', cwd))
		cwd = os.path.join(os.getcwd(), "main.py "+filepath+filename+" 'LSTM' 'GPU'")
		os.system('{} {}'.format('python3', cwd))
