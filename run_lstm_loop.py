import os
import sys
filename = "'data/synth_accum/accum1'"
if len(sys.argv)>1:
    filename = sys.argv[1]
cwd = os.path.join(os.getcwd(), "main.py "+filename+" 'LSTM' 'GPU'")
for i in range(10):
    print(i)
    os.system('{} {}'.format('python3', cwd))
