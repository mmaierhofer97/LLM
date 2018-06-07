import os
import sys
filenames = ["'data/synth_accum/accum_nonunif0.25'","'data/synth_accum/accum_nonunif0.0625'","'data/synth_accum/accum_nonunif1'","'data/synth_accum/accum_nonunif4'","'data/synth_accum/accum_nonunif16'"]
for filename in filenames:
  cwd = os.path.join(os.getcwd(), "main.py "+filename+" 'LLM' 'GPU'")
  for i in range(10):
      print(i)
      os.system('{} {}'.format('python3', cwd))
  cwd = os.path.join(os.getcwd(), "main.py "+filename+" 'LSTM' 'GPU'")
  for i in range(10):
      print(i)
      os.system('{} {}'.format('python3', cwd))
