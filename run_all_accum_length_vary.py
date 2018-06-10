import os
import sys
filenames = ["'data/synth_accum/accum_length_vary0'"]#,"'data/synth_accum/accum_length_vary1'","'data/synth_accum/accum_length_vary2'","'data/synth_accum/accum_length_vary3'","'data/synth_accum/accum_length_vary4'","'data/synth_accum/accum_length_vary5'"]
for i in range(1):
	print(i)
	for filename in filenames:
		cwd = os.path.join(os.getcwd(), "main.py "+filename+" 'LLM' 'GPU'")
		os.system('{} {}'.format('python3', cwd))
		cwd = os.path.join(os.getcwd(), "main.py "+filename+" 'LSTM' 'GPU'")
		os.system('{} {}'.format('python3', cwd))
