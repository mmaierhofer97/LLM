import os
import sys
filepath = "'data/synth_hawkes/hawkes3"
filename = "100"
if len(sys.argv)>1:
	filename = sys.argv[1]
for i in range(10):
	cwd = os.path.join(os.getcwd(), "gen_synth_hawkes3.py "+filename)
	os.system('{} {}'.format('python3', cwd))
	print(i)
	cwd = os.path.join(os.getcwd(), "main.py "+filepath+filename+"' 'LLM' 'GPU' 'PRED' 'FALSE'")
	os.system('{} {}'.format('python3', cwd))
	cwd = os.path.join(os.getcwd(), "main.py "+filepath+filename+"' 'LSTM' 'GPU' 'PRED' 'FALSE'")
	os.system('{} {}'.format('python3', cwd))
