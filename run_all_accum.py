import os
import sys
filepath = "'data/synth_accum/accum"
filenames = ["16'","1'","4'","0.0625'","0.25'"]
for i in range(10):
	print(i)
	for filename in filenames:
		cwd = os.path.join(os.getcwd(), "main.py "+filepath+filename+" 'LLM' 'GPU'")
		os.system('{} {}'.format('python3', cwd))
		cwd = os.path.join(os.getcwd(), "main.py "+filepath+filename+" 'LSTM' 'GPU'")
		os.system('{} {}'.format('python3', cwd))
